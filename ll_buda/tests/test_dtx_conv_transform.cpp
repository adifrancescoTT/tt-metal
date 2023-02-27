#include <algorithm>
#include <functional>
#include <random>
#include "ll_buda/impl/dtx/dtx.hpp"
#include "ll_buda/impl/dtx/dtx_passes.hpp"
#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// 1. Host writes data to buffer in DRAM
// 2. Host generates DTX transformation node -> Producer : 1D vector sized 64 -> Consumer : 1D vector of same size but flips first 32 elements with last 32 elements
// 2. dram_to_l1_copy_with_address_map kernel on logical core {0, 0} BRISC copies data from buffer in step 1. to buffer in L1
// 3. Host reads from buffer written to in step 2.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////


        DataTransformations * dtx = new DataTransformations();
        TransformationNode * node0 = new TransformationNode("producer", 1);
        node0->groups[0]->shape = {1,64,10,10};
        dtx->transformations.push_back(node0);
        pass &= conv3d_to_channels_last_transformation(dtx);
        //pass &= tilize_and_store(dtx, {0,1});
        dtx->print();
        DataTransformations * dtx_full = reverse_transformations(dtx);
        dtx_full->print();
        // 3x3 conv, stride=1, padding=0
        pass &= conv3d_to_matrix_transformation(dtx_full, {3,3,1,1,0,0});
        dtx_full->print();
        pass &= tilize_and_store(dtx_full, {1,0});
        dtx_full->print();
        //exit(0);
        pass &= collapse_transformations(dtx_full);
        dtx_full->print();
        exit(0);

        pass &= generate_transfer_addresses_tiled_data(dtx_full);

        // copy transfer addresses into a vector
        std::vector<uint32_t> address_map;
        for(auto transfer : dtx_full->transformations.back()->groups[0]->transfers){
            address_map.push_back(transfer->src_address*2); // 2 for bfloat16
            address_map.push_back(transfer->dst_address*2);
            address_map.push_back(transfer->size*2);
        }
        std::cout << "Address Map - " << std::endl;
        for(auto i = 0; i < address_map.size(); i+=3) {
            std::cout << "Source address - " << address_map[i];
            std::cout << ", Destination address - " << address_map[i+1];
            std::cout << ", Size to transfer in bytes - " << address_map[i+2] << std::endl;
        }
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair core = {0, 0};

        uint32_t dram_buffer_size = 2 * 64 * 64;
        uint32_t input_dram_buffer_addr = 0;
        uint32_t l1_buffer_addr = 400 * 1024;
        uint32_t address_map_l1_addr = 500 * 1024;
        int dram_channel = 0;

        auto input_dram_buffer = ll_buda::CreateDramBuffer(dram_channel, dram_buffer_size, input_dram_buffer_addr);

        auto l1_b0 = ll_buda::CreateL1Buffer(program, core, dram_buffer_size, l1_buffer_addr);

        auto input_dram_noc_xy = input_dram_buffer->noc_coordinates(device);

        auto dram_to_l1_copy_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/dram_to_l1_copy_with_address_map.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto input_vector = unpack_uint32_vec_into_bfloat16_vec(input_vec);
        pass &= ll_buda::WriteToDeviceDRAM(device, input_dram_buffer, input_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteToDeviceL1(device, core, address_map, address_map_l1_addr);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            dram_to_l1_copy_kernel,
            core,
            {input_dram_buffer_addr,
            (std::uint32_t)input_dram_noc_xy.x,
            (std::uint32_t)input_dram_noc_xy.y,
            l1_buffer_addr,
            address_map_l1_addr,
            (std::uint32_t) address_map.size()});

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceL1(device, core, l1_buffer_addr, result_vec, dram_buffer_size);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_vector = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        pass &= ll_buda::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
