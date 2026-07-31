// The contents of this file are in the public domain.
// See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the static vision tower of the Dlib
    transformer stack, and the way it is validated.

    A multimodal model ships two containers: the decoder, and a second one conventionally
    named mmproj-*.gguf holding the vision tower with the projector that brings its output
    into the decoder's embedding space. This library reads that second container in two
    different ways.

    runtime_vision_encoder is shape-dynamic: it compiles once and adapts to whatever
    geometry the file declares. It is the right tool for inference over models whose shape
    is only known at run time.

    vision_transformer_config is the same encoder expressed as a Dlib network type. Its
    geometry is fixed at compile time, which costs a recompilation per shape, and buys
    three things the dynamic path cannot give: the weights live in the network archive
    rather than in a separate file, the gradients flow, so the tower can be fine-tuned or
    trained from scratch, and every facility that applies to a Dlib network applies to it.

    Two implementations of one function invite one question: do they agree. This program
    answers it. It loads both from the same container, runs both on the same prepared
    image, and compares the results value by value. They are expected to match exactly,
    not approximately: both drive the same tensor primitives in the same order, so any
    difference at all would mean the weights were mapped wrong somewhere.

    Run it with:
        slm_vision_tower_ex --mmproj mmproj-SmolVLM-256M-Instruct-f16.gguf
        slm_vision_tower_ex --mmproj mmproj-SmolVLM-256M-Instruct-f16.gguf --image photo.png

    Without --image a deterministic synthetic picture is used, which makes the run
    reproducible and needs no file on disk.

    The geometry compiled in below is that of SmolVLM-256M. For a tower of another shape,
    change the vision_transformer_config arguments to the values --mmproj reports and
    rebuild: the program checks the two agree before loading anything.
*/

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <dlib/cmd_line_parser.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io/gguf_reader.h>
#include <dlib/data_io/gguf_vision_spec.h>
#include <dlib/data_io/gguf_vision_loader.h>

using namespace std;
using namespace dlib;

/* Geometry of the tower this build serves. Every value is checked against the container
   at load time, so a mismatch is reported rather than silently producing nonsense. */
using tower = vision_transformer_config<
    512,    // image size
    16,     // patch size
    768,    // width
    12,     // layers
    12,     // attention heads
    3072,   // feed-forward hidden size
    4,      // pixel shuffle factor
    576     // projection width, which is the decoder's embedding width
>;

/* A picture that needs no file: a smooth pattern in the three channels, identical from one
   run to the next, so the comparison below is reproducible anywhere. */
static void make_synthetic_image(matrix<rgb_pixel>& img, long side)
{
    img.set_size(side, side);
    for (long r = 0; r < side; ++r)
    {
        for (long c = 0; c < side; ++c)
        {
            img(r, c) = rgb_pixel(
                static_cast<unsigned char>((r * 7 + c * 3) % 256),
                static_cast<unsigned char>((r * c) % 256),
                static_cast<unsigned char>((r + 2 * c) % 256));
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("mmproj", "Projector container holding the vision tower", 1);
        parser.add_option("image", "Image to encode; a synthetic one is used without it", 1);
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);

        if (parser.option("h") || !parser.option("mmproj"))
        {
            cout << "Compare the static vision tower with the shape-dynamic encoder.\n\n";
            parser.print_options();
            cout << "Example:\n  " << argv[0]
                 << " --mmproj mmproj-SmolVLM-256M-Instruct-f16.gguf\n";
            return 0;
        }

        const string path = parser.option("mmproj").argument();
        gguf_reader g(path);
        const vision_spec spec = detect_vision(g);
        cout << describe(spec) << "\n";

        const vision_compat_result report = check_vision_compatibility(spec, g);
        for (const string& n : report.notes)    cout << "note: " << n << "\n";
        for (const string& b : report.blockers) cerr << "BLOCKER: " << b << "\n";
        if (!report.usable())
        {
            cerr << "This projector cannot be served by the vision path.\n";
            return 1;
        }

        // The image, prepared exactly once: the two encoders must see the same pixels.
        matrix<rgb_pixel> img;
        if (parser.option("image"))
        {
            load_image(img, parser.option("image").argument());
            cout << "Image              : " << parser.option("image").argument()
                 << " (" << img.nc() << "x" << img.nr() << ")\n";
        }
        else
        {
            make_synthetic_image(img, spec.image_size);
            cout << "Image              : synthetic, " << img.nc() << "x" << img.nr() << "\n";
        }

        runtime_vision_encoder encoder;
        cout << "\nLoading the shape-dynamic encoder...\n";
        encoder.load(g, spec);

        resizable_tensor prepared;
        encoder.prepare_image(img, prepared);

        auto started = chrono::steady_clock::now();
        const tensor& produced = encoder.encode(prepared);
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - started).count();

        /* Kept aside: the static tower is about to run and the reference must survive it. */
        resizable_tensor reference;
        reference.copy_size(produced);
        memcpy(reference, produced);
        cout << "Dynamic encoder    : " << reference.num_samples() << " x "
             << reference.size() / static_cast<size_t>(reference.num_samples())
             << " in " << elapsed << " ms\n";

        cout << "\n" << tower::model_info::describe() << "\n";
        tower::network_type net;
        gguf_reader again(path);
        cout << "Loading the static tower...\n";
        import_gguf_vision_weights(net, again, spec, tower());

        started = chrono::steady_clock::now();
        const tensor& out = net.forward(prepared);
        elapsed = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - started).count();
        cout << "Static tower       : " << out.num_samples() << " x "
             << out.size() / static_cast<size_t>(out.num_samples())
             << " in " << elapsed << " ms\n\n";

        if (out.size() != reference.size())
        {
            cerr << "The two encoders do not even agree on the number of values.\n";
            return 1;
        }

        double max_abs = 0, sum_sq = 0, ref_sq = 0;
        for (size_t i = 0; i < out.size(); ++i)
        {
            const double d = static_cast<double>(out.host()[i]) - reference.host()[i];
            max_abs = std::max(max_abs, std::abs(d));
            sum_sq += d * d;
            ref_sq += static_cast<double>(reference.host()[i]) * reference.host()[i];
        }
        cout << scientific << setprecision(3);
        cout << "Largest difference : " << max_abs << "\n";
        cout << "Relative L2        : " << std::sqrt(sum_sq / std::max(1e-30, ref_sq)) << "\n";
        cout << defaultfloat;

        if (max_abs == 0.0)
            cout << "\nThe two paths agree exactly.\n";
        else if (max_abs < 1e-4)
            cout << "\nThe two paths agree to within rounding.\n";
        else
            cout << "\nThe two paths DISAGREE: the weight mapping is wrong somewhere.\n";

        return max_abs < 1e-4 ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        cerr << "\nFATAL: " << e.what() << "\n";
        return 1;
    }
}
