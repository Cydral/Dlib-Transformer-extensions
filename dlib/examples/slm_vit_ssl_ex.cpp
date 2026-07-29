// The contents of this file are in the public domain.
// See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating how to train the vision transformer of the Dlib
    transformer stack without labels, and what that buys.

    The objective is Barlow Twins. Two augmented views of the same image go through the
    same tower, a projector maps both to a wider space, and the loss drives the
    cross-correlation matrix of the two projections towards the identity: each coordinate
    should agree between the two views, and no two coordinates should say the same thing.
    Agreement teaches invariance to the augmentation, decorrelation stops the network from
    collapsing to a constant, which is what every method of this family has to prevent.

        template <typename SUBNET> using projector = fc<128, relu<bn_fc<fc<512, SUBNET>>>>;
        using train_net = loss_barlow_twins<projector<tower::image_embedding>>;
        using feats_net = loss_metric<tower::image_embedding>;

    The projector is thrown away afterwards. What is kept is image_embedding, the tower
    with its patch pooling, which turns a picture into one vector. That is what gets
    evaluated here, by a nearest-neighbour classifier over the labels: the labels never
    reach the training, only the measurement.

    The payoff for this library specifically. A tower trained this way is a
    vision_transformer_config::network_type like any other, so it can be handed to a fusion
    layer as an already-trained encoder:

        modality_fusion_<tower::network_type, ...> fusion;
        deserialize(tower_file) >> fusion.get_encoder();

    which is the first stage of the usual multimodal recipe, learned here without a single
    caption. The tower saved by this program is written in exactly that form, over an
    image input rather than a tensor input; the two differ only in their bottom layer, and
    the note printed at the end says how to move between them.

    Run it with:
        slm_vit_ssl_ex --data /path/to/cifar-10-batches-bin
        slm_vit_ssl_ex --data /path/to/cifar-10-batches-bin --epochs 40 --lambda 0.005

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

/* The same geometry as the supervised example, so the two can be read against each other.
   The input layer is the pair variant: Barlow Twins needs two views per sample and the
   loss expects them stacked, first views then second views. */
template <typename INPUT>
using tower_over = vision_transformer_config<32, 4, 128, 6, 4, 512, 1, 128, INPUT>;

using pair_tower = tower_over<input_rgb_image_pair>;
using solo_tower = tower_over<input_rgb_image_sized<32>>;

/* Wider than the embedding, as this family of methods wants: the decorrelation term has
   more coordinates to spread information over than the representation itself. */
template <typename SUBNET> using projector = fc<128, relu<bn_fc<fc<512, SUBNET>>>>;

using train_net = loss_barlow_twins<projector<pair_tower::image_embedding>>;
using feats_net = loss_metric<solo_tower::image_embedding>;

/* The augmentation is the objective. Everything the two views share is what the tower
   will learn to represent, and everything they do not is what it will learn to ignore, so
   this function decides what the representation means more than any hyperparameter does. */
static matrix<rgb_pixel> augment(const matrix<rgb_pixel>& image, bool flip, dlib::rand& rnd)
{
    matrix<rgb_pixel> crop;
    const auto rect = rectangle(image.nc() - 1, image.nr() - 1);
    const double scale = 0.6 + 0.4 * rnd.get_random_double();
    const long side = std::max<long>(8, static_cast<long>(image.nr() * scale));
    const long r = rnd.get_integer(std::max<long>(1, image.nr() - side));
    const long c = rnd.get_integer(std::max<long>(1, image.nc() - side));
    const rectangle window = rectangle(c, r, c + side - 1, r + side - 1).intersect(rect);

    matrix<rgb_pixel> region = subm(image, window);
    crop.set_size(image.nr(), image.nc());
    resize_image(region, crop);
    if (flip) crop = fliplr(crop);

    // Colour jitter, so that the representation does not reduce to average hue.
    if (rnd.get_random_double() < 0.8)
    {
        const double gain = 0.7 + 0.6 * rnd.get_random_double();
        const double bias = -20.0 + 40.0 * rnd.get_random_double();
        for (long y = 0; y < crop.nr(); ++y)
            for (long x = 0; x < crop.nc(); ++x)
            {
                const rgb_pixel p = crop(y, x);
                crop(y, x) = rgb_pixel(
                    static_cast<unsigned char>(put_in_range(0, 255, gain * p.red + bias)),
                    static_cast<unsigned char>(put_in_range(0, 255, gain * p.green + bias)),
                    static_cast<unsigned char>(put_in_range(0, 255, gain * p.blue + bias)));
            }
    }
    if (rnd.get_random_double() < 0.2)
    {
        matrix<rgb_pixel> grey(crop.nr(), crop.nc());
        for (long y = 0; y < crop.nr(); ++y)
            for (long x = 0; x < crop.nc(); ++x)
            {
                const rgb_pixel p = crop(y, x);
                const unsigned char v = static_cast<unsigned char>(
                    0.299 * p.red + 0.587 * p.green + 0.114 * p.blue);
                grey(y, x) = rgb_pixel(v, v, v);
            }
        crop = grey;
    }
    return crop;
}

/* Nearest-neighbour accuracy over the embeddings, which is the standard way to read what a
   self-supervised representation has learned: no head is fitted, so the number measures
   the representation and not a classifier trained on top of it. */
static double knn_accuracy(feats_net& net, const std::vector<matrix<rgb_pixel>>& train,
    const std::vector<unsigned long>& train_labels,
    const std::vector<matrix<rgb_pixel>>& test,
    const std::vector<unsigned long>& test_labels)
{
    const std::vector<matrix<float, 0, 1>> ref = net(train, 128);
    const std::vector<matrix<float, 0, 1>> query = net(test, 128);

    size_t right = 0;
    for (size_t i = 0; i < query.size(); ++i)
    {
        double best = std::numeric_limits<double>::max();
        size_t at = 0;
        for (size_t j = 0; j < ref.size(); ++j)
        {
            const double d = length_squared(query[i] - ref[j]);
            if (d < best) { best = d; at = j; }
        }
        if (train_labels[at] == test_labels[i]) ++right;
    }
    return query.empty() ? 0.0 : static_cast<double>(right) / query.size();
}

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("data", "Directory holding the CIFAR-10 binary distribution", 1);
        parser.add_option("epochs", "Passes over the training set (default: 30)", 1);
        parser.add_option("batch-size", "Image pairs per step (default: 64)", 1);
        parser.add_option("learning-rate", "Initial learning rate (default: 1e-3)", 1);
        parser.add_option("lambda", "Weight of the decorrelation term (default: 0.005)", 1);
        parser.add_option("probe-size", "Images used by the nearest-neighbour probe (default: 5000)", 1);
        parser.add_option("out", "Where to write the trained tower (default: vit_ssl_tower.dat)", 1);
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);

        if (parser.option("h") || !parser.option("data"))
        {
            cout << "Train a vision transformer without labels, by Barlow Twins.\n\n";
            parser.print_options();
            cout << "Example:\n  " << argv[0] << " --data ./cifar-10-batches-bin\n";
            return 0;
        }

        std::vector<matrix<rgb_pixel>> train_images, test_images;
        std::vector<unsigned long> train_labels, test_labels;
        cout << "Loading CIFAR-10 from " << parser.option("data").argument() << " ...\n";
        load_cifar_10_dataset(parser.option("data").argument(),
            train_images, train_labels, test_images, test_labels);
        cout << "  training : " << train_images.size() << " images (labels unused)\n"
             << "  testing  : " << test_images.size() << " images\n";

        cout << "\n" << pair_tower::model_info::describe() << "\n";

        const double lambda = get_option(parser, "lambda", 0.005);
        train_net net((loss_barlow_twins_(static_cast<float>(lambda))));

        const long epochs = get_option(parser, "epochs", 30);
        const size_t batch = static_cast<size_t>(get_option(parser, "batch-size", 64));
        const double lr = get_option(parser, "learning-rate", 1e-3);

        dnn_trainer<train_net, adamw> trainer(net, adamw(0.05f, 0.9f, 0.999f));
        trainer.set_learning_rate(lr);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(batch);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_iterations_without_progress_threshold(3000);
        trainer.be_verbose();
        trainer.set_synchronization_file("vit_ssl_sync", std::chrono::minutes(10));

        dlib::rand rnd(std::time(nullptr));
        std::vector<std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> views;

        const auto started = std::chrono::steady_clock::now();
        for (long epoch = 0; epoch < epochs; ++epoch)
        {
            std::vector<size_t> order(train_images.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            for (size_t i = order.size(); i > 1; --i)
                std::swap(order[i - 1], order[rnd.get_integer(static_cast<int>(i))]);

            for (size_t i = 0; i + batch <= order.size(); i += batch)
            {
                views.clear();
                for (size_t j = 0; j < batch; ++j)
                {
                    const matrix<rgb_pixel>& image = train_images[order[i + j]];
                    views.emplace_back(augment(image, false, rnd), augment(image, true, rnd));
                }
                trainer.train_one_step(views.begin(), views.end());
            }
            trainer.get_net(force_flush_to_disk::no);
            cout << "  epoch " << (epoch + 1) << "/" << epochs
                 << "  learning rate " << trainer.get_learning_rate()
                 << "  average loss " << trainer.get_average_loss() << "\n";
            trainer.clear_average_loss();
            if (trainer.get_learning_rate() < trainer.get_min_learning_rate()) break;
        }
        trainer.get_net();
        net.clean();

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - started).count();
        cout << "\ntrained in " << elapsed << " s\n";

        /* Drop the projector and keep the tower. Constructing the feature network from
           the layer where the tower begins is Dlib's own idiom for this, and it works
           across the two input layers because everything above them is the same sequence:
           the pair input feeds two views, the single input one picture, and no layer in
           between knows the difference. Four layers of projector sit above the tower, plus
           the loss at index zero. */
        feats_net feats(layer<5>(net));

        const size_t probe = static_cast<size_t>(get_option(parser, "probe-size", 5000));
        const std::vector<matrix<rgb_pixel>> ref(train_images.begin(),
            train_images.begin() + std::min(probe, train_images.size()));
        const std::vector<unsigned long> ref_labels(train_labels.begin(),
            train_labels.begin() + std::min(probe, train_labels.size()));
        const std::vector<matrix<rgb_pixel>> query(test_images.begin(),
            test_images.begin() + std::min(probe / 5, test_images.size()));
        const std::vector<unsigned long> query_labels(test_labels.begin(),
            test_labels.begin() + std::min(probe / 5, test_labels.size()));

        cout << "nearest-neighbour accuracy over the embeddings : "
             << knn_accuracy(feats, ref, ref_labels, query, query_labels)
             << "  (labels used only here, never in training)\n";

        const std::string out_path = get_option(parser, "out", std::string("vit_ssl_tower.dat"));
        serialize(out_path) << feats;
        cout << "written to " << out_path << "\n\n"
             << "This tower is a vision_transformer_config::network_type over an image\n"
             << "input. To hand it to a fusion layer, which drives its tower with a prepared\n"
             << "tensor, keep the same geometry and name the tensor input instead: the two\n"
             << "differ only in their bottom layer, and the layers above deserialize into\n"
             << "one another unchanged.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "\nFATAL: " << e.what() << "\n";
        return 1;
    }
}
