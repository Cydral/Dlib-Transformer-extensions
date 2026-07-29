// The contents of this file are in the public domain.
// See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the vision transformer of the Dlib
    transformer stack as a plain image backbone, trained from scratch on CIFAR-10.

    There is no language model here and no pretrained weights. The point is that the tower
    built for multimodal work is a backbone in its own right: name an image input instead
    of a tensor input, pool over the patches, put a head on top, and it trains like any
    other Dlib network.

        using tower = dlib::vision_transformer_config<
            32, 4, 128, 6, 4, 512, 1, 128, dlib::input_rgb_image_sized<32>>;
        using net_type = dlib::loss_multiclass_log<tower::classifier<10>>;

    Two lines. Everything a vision transformer needs is behind them: the patch embedding
    convolution, the learned position table, the bidirectional attention blocks, the final
    normalization, and the pooling that turns a bag of patch vectors into one vector per
    image.

    Three choices in that configuration are worth explaining, because a transformer trained
    on small images from scratch is easy to get wrong.

    The patch size is 4 rather than 16. On a 32 by 32 image, patches of 16 leave four
    tokens, which is not a sequence; patches of 4 leave sixty-four, which is.

    The shuffle factor is 1, which disables the pixel shuffle. That fold exists to shorten
    the sequence before a projector feeds a language model, and shortening a sequence of
    sixty-four is the opposite of what a classifier wants.

    The projection width equals the tower width, so the projector is a square matrix that
    the classification head sits on. Nothing forces those two to differ outside multimodal
    use, where the projector's job is to match the decoder's embedding width.

    Run it with:
        slm_vit_classify_ex --data /path/to/cifar-10-batches-bin
        slm_vit_classify_ex --data /path/to/cifar-10-batches-bin --epochs 30 --batch-size 64

    The dataset is the binary CIFAR-10 distribution from the University of Toronto:
    https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
*/

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;

/* The backbone. Small enough to train on a workstation, large enough that the attention
   blocks have something to do. */
using tower = vision_transformer_config<
    32,     // image size
    4,      // patch size: 64 patches, which is a sequence worth attending over
    128,    // width
    6,      // blocks
    4,      // heads
    512,    // feed-forward hidden size
    1,      // no pixel shuffle: that fold serves a language model, not a classifier
    128,    // projection width, equal to the tower width outside multimodal use
    input_rgb_image_sized<32>>;

using net_type = loss_multiclass_log<tower::classifier<10>>;

/* Random crops with a flip, which is the whole augmentation this task needs. A transformer
   has no convolutional inductive bias, so it leans on augmentation more heavily than a
   residual network of the same size; without this, the training accuracy runs away from
   the test accuracy within a few epochs. */
static matrix<rgb_pixel> augment(const matrix<rgb_pixel>& image, dlib::rand& rnd)
{
    matrix<rgb_pixel> padded, out;
    const long pad = 4;
    padded.set_size(image.nr() + 2 * pad, image.nc() + 2 * pad);
    assign_all_pixels(padded, rgb_pixel(0, 0, 0));
    set_subm(padded, rectangle(pad, pad, pad + image.nc() - 1, pad + image.nr() - 1)) = image;

    const long r = rnd.get_integer(2 * pad + 1);
    const long c = rnd.get_integer(2 * pad + 1);
    out = subm(padded, rectangle(c, r, c + image.nc() - 1, r + image.nr() - 1));
    if (rnd.get_random_double() < 0.5) out = fliplr(out);
    return out;
}

static double accuracy(net_type& net, const std::vector<matrix<rgb_pixel>>& images,
    const std::vector<unsigned long>& labels, size_t chunk = 256)
{
    size_t right = 0;
    for (size_t i = 0; i < images.size(); i += chunk)
    {
        const size_t upto = std::min(i + chunk, images.size());
        std::vector<matrix<rgb_pixel>> batch(images.begin() + i, images.begin() + upto);
        const std::vector<unsigned long> got = net(batch);
        for (size_t j = 0; j < got.size(); ++j)
            if (got[j] == labels[i + j]) ++right;
    }
    return images.empty() ? 0.0 : static_cast<double>(right) / images.size();
}

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("data", "Directory holding the CIFAR-10 binary distribution", 1);
        parser.add_option("epochs", "Passes over the training set (default: 20)", 1);
        parser.add_option("batch-size", "Images per step (default: 64)", 1);
        parser.add_option("learning-rate", "Initial learning rate (default: 1e-3)", 1);
        parser.add_option("out", "Where to write the trained tower (default: vit_cifar10.dat)", 1);
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);

        if (parser.option("h") || !parser.option("data"))
        {
            cout << "Train a vision transformer from scratch on CIFAR-10.\n\n";
            parser.print_options();
            cout << "Example:\n  " << argv[0] << " --data ./cifar-10-batches-bin\n";
            return 0;
        }

        std::vector<matrix<rgb_pixel>> train_images, test_images;
        std::vector<unsigned long> train_labels, test_labels;
        cout << "Loading CIFAR-10 from " << parser.option("data").argument() << " ...\n";
        load_cifar_10_dataset(parser.option("data").argument(),
            train_images, train_labels, test_images, test_labels);
        cout << "  training : " << train_images.size() << " images\n"
             << "  testing  : " << test_images.size() << " images\n";

        cout << "\n" << tower::model_info::describe() << "\n";

        net_type net;
        /* One forward before counting: a Dlib network allocates its parameters on the
           first pass through it, so a count taken before that reads zero. */
        {
            std::vector<matrix<rgb_pixel>> one(1, train_images[0]);
            net.subnet()(one.begin(), one.end());
            cout << "\nparameters : " << count_parameters(net) << "\n";
        }

        const long epochs = get_option(parser, "epochs", 20);
        const size_t batch = static_cast<size_t>(get_option(parser, "batch-size", 64));
        const double lr = get_option(parser, "learning-rate", 1e-3);

        /* AdamW rather than plain SGD. A transformer trained from scratch on this little
           data is sensitive to the step size, and the decoupled decay keeps the attention
           projections from growing without bounds. */
        dnn_trainer<net_type, adamw> trainer(net, adamw(0.05f, 0.9f, 0.999f));
        trainer.set_learning_rate(lr);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(batch);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_iterations_without_progress_threshold(2000);
        trainer.be_verbose();
        trainer.set_synchronization_file("vit_cifar10_sync", std::chrono::minutes(10));

        dlib::rand rnd(std::time(nullptr));
        std::vector<matrix<rgb_pixel>> batch_images;
        std::vector<unsigned long> batch_labels;

        const auto started = std::chrono::steady_clock::now();
        for (long epoch = 0; epoch < epochs; ++epoch)
        {
            std::vector<size_t> order(train_images.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            for (size_t i = order.size(); i > 1; --i)
                std::swap(order[i - 1], order[rnd.get_integer(static_cast<int>(i))]);

            for (size_t i = 0; i + batch <= order.size(); i += batch)
            {
                batch_images.clear();
                batch_labels.clear();
                for (size_t j = 0; j < batch; ++j)
                {
                    const size_t k = order[i + j];
                    batch_images.push_back(augment(train_images[k], rnd));
                    batch_labels.push_back(train_labels[k]);
                }
                trainer.train_one_step(batch_images, batch_labels);
            }

            /* get_net() is also the synchronization barrier with the trainer's background
               thread, so the network is quiet while it is evaluated. */
            net_type& current = trainer.get_net(force_flush_to_disk::no);
            cout << "  epoch " << (epoch + 1) << "/" << epochs
                 << "  learning rate " << trainer.get_learning_rate()
                 << "  average loss " << trainer.get_average_loss()
                 << "  test accuracy " << accuracy(current, test_images, test_labels) << "\n";
            trainer.clear_average_loss();
            if (trainer.get_learning_rate() < trainer.get_min_learning_rate()) break;
        }
        trainer.get_net();
        net.clean();

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - started).count();
        cout << "\ntrained in " << elapsed << " s\n";
        cout << "final accuracy : training "
             << accuracy(net, train_images, train_labels) << ", testing "
             << accuracy(net, test_images, test_labels) << "\n";

        const std::string out_path = get_option(parser, "out", std::string("vit_cifar10.dat"));
        serialize(out_path) << net;
        cout << "written to " << out_path << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "\nFATAL: " << e.what() << "\n";
        return 1;
    }
}
