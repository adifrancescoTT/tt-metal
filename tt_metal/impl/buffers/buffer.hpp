// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include <map>
#include <optional>


namespace tt {

namespace tt_metal {

class Device;

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY
};


enum class TensorMemoryLayout {
    INTERLEAVED,
    SINGLE_BANK,
    HEIGHT_SHARDED,
    WIDTH_SHARDED,
    BLOCK_SHARDED,
};

enum class ShardOrientation {
    ROW_MAJOR,
    COL_MAJOR,
};


struct ShardSpec {
    CoreRangeSet shard_grid;
    std::array<uint32_t, 2> shard_shape;
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;
    bool halo = false;

    ShardSpec(const CoreRangeSet & core_sets_,
                    const std::array<uint32_t,2> & shard_shape_,
                    const ShardOrientation & shard_orientation_ = ShardOrientation::ROW_MAJOR,
                    const bool & halo_ = false):
                    shard_grid(core_sets_), shard_shape(shard_shape_),
                    shard_orientation(shard_orientation_), halo(halo_)
                    {;}

    const uint32_t num_cores() const { return this->shard_grid.num_cores(); }
    const uint32_t numel() const { return this->shard_shape[0] * this->shard_shape[1]; }
    tt::stl::reflection::Attributes attributes() const;

};


struct ShardSpecBuffer:  ShardSpec {
    std::array<uint32_t, 2> page_shape;
    std::array<uint32_t, 2 > tensor2d_size;
    ShardSpecBuffer(const CoreRangeSet & core_sets_,
                const std::array<uint32_t,2> & shard_shape_,
                const ShardOrientation & shard_orientation_,
                const bool & halo_,
                const std::array<uint32_t, 2> & page_shape,
                const std::array<uint32_t, 2> & tensor2d_shape
                ): ShardSpec(core_sets_, shard_shape_, shard_orientation_, halo_)
                {
                    this->page_shape = page_shape;
                    this-> tensor2d_size = tensor2d_shape;
                }
    ShardSpecBuffer(
            const ShardSpec & shard_spec,
            const std::array<uint32_t, 2> & page_shape,
            const std::array<uint32_t, 2> & tensor2d_shape
            ): ShardSpec(shard_spec)
            {
                this->page_shape = page_shape;
                this-> tensor2d_size = tensor2d_shape;
            }
};


bool is_sharded(const TensorMemoryLayout & layout);
class Buffer {
   public:
    Buffer() : device_(nullptr) {}

    Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
        const TensorMemoryLayout buffer_layout=TensorMemoryLayout::INTERLEAVED,
        std::optional<ShardSpecBuffer> shard_parameter = std::nullopt);

    Buffer(const Buffer &other);
    Buffer& operator=(const Buffer &other);

    Buffer(Buffer &&other);
    Buffer& operator=(Buffer &&other);

    ~Buffer();

    Device *device() const { return device_; }

    uint32_t size() const { return static_cast<uint32_t>(size_); }

    // Returns address of buffer in the first bank
    uint32_t address() const { return static_cast<uint32_t>(address_); }

    uint32_t page_size() const { return page_size_; }

    uint32_t num_pages() const { return this->size() / this->page_size(); }

    BufferType buffer_type() const { return buffer_type_; }

    TensorMemoryLayout buffer_layout() const { return buffer_layout_; }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    CoreCoord noc_coordinates(uint32_t bank_id) const;

    // returns NoC coordinates of first bank buffer is in
    CoreCoord noc_coordinates() const;

    uint64_t page_address(uint32_t bank_id, uint32_t page_index) const;

    uint64_t core_address(uint32_t core_id) const;

    CoreRangeSet shard_grid() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return shard_parameters_.value().shard_grid;
    }

    ShardOrientation shard_orientation() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return shard_parameters_.value().shard_orientation;
    }


    uint32_t shard_size() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        auto p_shape = this->page_shape();
        auto width_in_pages = shard_parameters_.value().shard_shape[0] / p_shape[0];
        auto height_in_pages = shard_parameters_.value().shard_shape[1] / p_shape[1];
        return width_in_pages * height_in_pages;
    }


    std::array<uint32_t, 2> page_shape() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return {shard_parameters_.value().page_shape[0], shard_parameters_.value().page_shape[1]};
    }


    std::array<uint32_t,2> shard_shape() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        auto p_shape = page_shape();
        return {shard_parameters_.value().shard_shape[0],
                shard_parameters_.value().shard_shape[1]};
    }

    std::array<uint32_t, 2> tensor2d_size() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return {shard_parameters_.value().tensor2d_size[0], shard_parameters_.value().tensor2d_size[1]};
    }


    std::vector<uint32_t> dev_page_to_core_mapping() const{
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return dev_page_to_core_mapping_;
    }

    std::vector<CoreCoord> all_cores() const{
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return all_cores_;
    }

    std::vector< std::vector<uint32_t> > core_host_page_indices() const{
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return core_host_page_indices_;
    }

    std::vector < uint32_t> core_bank_indices() const{
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return core_bank_indices_;
    }

    std::vector < uint32_t> dev_page_to_host_page_mapping() const{
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return dev_page_to_host_page_mapping_;
    }

    uint32_t num_cores() const{
        if(!is_sharded(this->buffer_layout_))
            return 1;
        else{
            auto num_pages = this->size()/this->page_size();
            auto shards_for_compute = num_pages/this->shard_size();
            return shards_for_compute;
        }
    }

    std::unordered_map<CoreCoord, uint32_t> core_to_core_id() const{
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return core_to_core_id_;
    }

    std::vector<uint32_t> host_pages_in_shard(uint32_t core_id) const
    {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return core_host_page_indices_[core_id];
    }

    std::vector<uint32_t> host_pages_in_shard(CoreCoord core) const
    {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        auto core_id = core_to_core_id_.at(core);
        return core_host_page_indices_[core_id];
    }

    std::vector<uint32_t> dev_pages_in_shard(const uint32_t & core_id) const
    {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        auto host_pages= core_host_page_indices_[core_id];
        std::vector<uint32_t> dev_pages;
        dev_pages.reserve(host_pages.size());
        for(auto host_page: host_pages){
            dev_pages.push_back(dev_page_to_host_page_mapping_[host_page]);
        }
        return dev_pages;
    }

    std::vector<uint32_t> dev_pages_in_shard(const CoreCoord & core) const
    {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        auto core_id = core_to_core_id_.at(core);
        return dev_pages_in_shard(core_id);
    }

    std::string get_shard_info() const;
    void print_shard_info() const;

    void log_shard_info() const;

   private:
    void allocate();

    void deallocate();
    friend void DeallocateBuffer(Buffer &buffer);

    Device *device_;
    uint64_t size_;                 // Size in bytes
    uint64_t address_;              // Address of buffer
    uint64_t page_size_;            // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type_;
    TensorMemoryLayout buffer_layout_;
    std::optional<ShardSpecBuffer> shard_parameters_;
    std::vector< CoreCoord> all_cores_;
    std::vector< uint32_t> core_bank_indices_;
    std::vector< std::vector<uint32_t> > core_host_page_indices_;
    std::vector<uint32_t> dev_page_to_core_mapping_;
    std::vector<uint32_t> dev_page_to_host_page_mapping_;
    std::unordered_map<CoreCoord, uint32_t> core_to_core_id_;
};

}  // namespace tt_metal

}  // namespace tt
