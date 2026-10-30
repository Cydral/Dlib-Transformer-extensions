// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_TENSOR_BACKEND_H_
#define ALBIREO_ENGINE_TENSOR_BACKEND_H_

#include <functional>
#include <memory>
#include <string>

#include "../common/json.h"
#include "offload.h"

namespace albireo
{
    class thread_pool;

    class tensor_device
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The device of an engine built on the tensors of dlib, and the products it
                takes from the processor runtime of every model the engine loads:
                    - a graphics processor of NVIDIA, when dlib is built with CUDA;
                    - the NPU and the integrated graphics processor of an Intel processor,
                      when dlib is built with OpenVINO (DLIB_USE_NPU);
                    - the processor itself otherwise, through the host code of dlib, a
                      configuration that serves to validate this code on any machine.

                A product runs on the device with a copy of its weight matrix in single
                precision, the element type of the tensors of dlib.  The copies are made
                the first time the device takes a matrix and kept while they fit in a
                budget shared by the models.  A model that lacks room takes it from the
                models without requests, the least recently used first, each giving the
                room asked and keeping its other copies; the copies that leave come back
                when their model needs them again.  The copies of a model with requests in
                progress or waiting stay, so that the requests of several models are served
                together, each model keeping the room it holds.  Beyond the budget, a matrix
                is expanded again at each product it is given (CUDA), or left to the
                processor (OpenVINO), until a model has no more requests or copies leave.

                Where a product runs is measured, shape by shape: the first products of a
                shape run on the device and on the processor in turn, the faster place keeps
                the shape, and the choice is checked again at growing intervals.  With
                OpenVINO, the routing of dlib makes these measures itself, against the
                processor kernels of the engine.

                On a CUDA device, the extended memory of dlib (V²RAM) lets the copies exceed
                the memory of the card: the blocks that do not fit move to the memory of the
                host or to a file, and come back ahead of their use.

            CONFIGURATION
                The object the engine receives at its creation (engine_api.h), of which this
                object reads:
                    device        CUDA: the index of the card, empty for the one with the
                                  most memory.  OpenVINO: the devices, "NPU", "GPU", "GPU.1"
                                  or a list such as "NPU,GPU".
                    v2ram         true (the default) to start the extended memory (CUDA).
                    v2ram_max_mb  the size of its file in megabytes, 0 for automatic.
                    offload_mb    the most the copies of the weights may occupy, in
                                  megabytes, 0 for automatic.
                    offload_min_tokens
                                  products over fewer tokens stay on the processor, 1 by
                                  default.
                    routing       "auto" (the default) measures where each shape runs
                                  faster, "device" sends every product to the device and
                                  "host" none, for tests and diagnostics.
                    cache_dir     the folder of the compiled graphs (OpenVINO) and of the
                                  file of the extended memory.
                    benchmark     true when the engine is being measured: the graphs then
                                  compile on their first use instead of in the background.
        !*/
    public:
        typedef std::function<void(int level, const std::string& message)> log_function;

        tensor_device(const json& config, thread_pool& pool, log_function log);
        ~tensor_device();

        tensor_device(const tensor_device&) = delete;
        tensor_device& operator=(const tensor_device&) = delete;

        // The offload of one model, named for the messages; it must be destroyed before
        // this object.
        std::unique_ptr<matmul_offload> make_offload(const std::string& model = std::string());

        // The counters of the device, for the statistics of the engine.
        json stats() const;

        // The counters of the extended memory, or null when it does not run.
        json memory() const;

        /*!
            ensures
                - Returns the devices this build can use on this machine, in the form of the
                  "devices" of engine_api.h, and sets reason when there is none, with its
                  code and the values of the code in args (engine_api.h, "reason_code").
                - Fills backend with the name of the backend ("cuda", "openvino", "dlib")
                  and runtime with the version of its runtime, when known.
        !*/
        static json describe(std::string& backend, std::string& runtime, std::string& reason, std::string& code,
            json& args);

    private:
        struct impl;
        std::unique_ptr<impl> impl_;
        friend class tensor_offload;
    };

}

#endif // ALBIREO_ENGINE_TENSOR_BACKEND_H_
