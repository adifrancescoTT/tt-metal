#pragma once

#include <memory>

#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt {

namespace tt_metal {

// Represents all cores within range specified by the two cores
using CoreRange = std::pair<tt_xy_pair, tt_xy_pair>;
using CoreBlocks = std::vector<std::variant<tt_xy_pair, CoreRange>>;

template<class... Ts> struct overloaded_core : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded_core(Ts...) -> overloaded_core<Ts...>;

// Fwd declares
class CircularBuffer;
class DramBuffer;
class InterleavedDramBuffer;
class L1Buffer;
class Program;

// A physical PCIexpress Tenstorrent device
class Device {
   public:

    friend void tt_gdb(Device* device, int chip_id, const vector<tt_xy_pair> cores, vector<string> ops);
    Device(tt::ARCH arch, int pcie_slot) : arch_(arch), cluster_(nullptr), pcie_slot_(pcie_slot), closed_(false) {}

    ~Device();

    // TODO: Add copy/move semantics
    Device(const Device &other) { }
    Device& operator=(const Device &other) { return *this; }

    Device(Device &&other) { }
    Device& operator=(Device &&other) { return *this; }

    tt::ARCH arch() const { return arch_; }

    int pcie_slot() const { return pcie_slot_; }

    tt_cluster *cluster() const;  // Need to access cluster in llrt APIs

    int num_dram_banks() const;

    uint32_t l1_size() const;

    tt_xy_pair logical_grid_size() const;

    tt_xy_pair compute_and_storage_grid_size() const;

    tt_xy_pair worker_core_from_logical_core(const tt_xy_pair &logical_core) const;

    std::vector<tt_xy_pair> worker_cores_from_logical_cores(const std::vector<tt_xy_pair> &logical_cores);

   private:
    bool cluster_is_initialized() const { return cluster_ != nullptr; }

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize();
    friend bool InitializeDevice(Device *device);
    void initialize_cluster();
    void initialize_allocator();

    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    // Interfaces to memory manager
    uint32_t allocate_dram_buffer(int dram_channel, uint32_t size_in_bytes);
    uint32_t allocate_dram_buffer(int dram_channel, uint32_t size_in_bytes, uint32_t address);
    void free_dram_buffer(int dram_channel, uint32_t address);
    uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_in_bytes);
    uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_in_bytes, uint32_t address);
    void free_l1_buffer(const tt_xy_pair &logical_core, uint32_t address);
    uint32_t allocate_sysmem_buffer(uint32_t size_in_bytes);
    void free_sysmem_buffer(uint32_t address);
    uint32_t address_for_interleaved_dram_buffer(const std::map<int, uint32_t> &size_in_bytes_per_bank);
    uint32_t address_for_l1_buffers_across_core_range(const CoreRange &logical_core_range, uint32_t size_in_bytes);
    friend class DramBuffer;
    friend class InterleavedDramBuffer;
    friend class L1Buffer;
    // friend class SysMemBuffer;
    friend std::vector<L1Buffer *> CreateL1Buffers(Program *program, Device *device, const CoreRange &core_range, uint32_t size_in_bytes);
    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program *program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format
    );

    static constexpr TargetDevice target_type_ = TargetDevice::Silicon;
    tt::ARCH arch_;
    tt_cluster *cluster_;
    int pcie_slot_;
    std::unique_ptr<Allocator> allocator_;
    bool closed_;
};

}  // namespace tt_metal

}  // namespace tt
