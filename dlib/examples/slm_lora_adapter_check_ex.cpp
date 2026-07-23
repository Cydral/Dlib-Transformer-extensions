/*!
    @file slm_lora_adapter_check_ex.cpp
    @brief Numerical validation of the low-rank adaptation core.

    low_rank_adapter is the one piece of the fine-tuning path whose correctness cannot be
    read off the output of a training run: a wrong gradient does not crash, it produces a
    loss curve that goes down slowly and a model that is merely disappointing. This
    program settles the question before the core is wired into the attention layer, by
    checking it against a double-precision reference written straight from the definition
    of each method.

    Four checks, run for LoRA and for DoRA:

      inertness  a freshly initialized adapter must reproduce the base projection bit for
                 bit, not approximately. B starts at zero and, for DoRA, the magnitudes
                 start at the column norms of the base, so the column factors are exactly
                 one. Any drift here means the initialization is wrong, and every later
                 comparison with an unadapted run would be meaningless.
      forward    the adapted output must equal x times the merged weight. This is what
                 justifies never forming that weight.
      gradient   every parameter gradient, and the input gradient, against central finite
                 differences of the reference. The magnitude path of DoRA is exact here,
                 not detached, so the check applies to it like to the rest.
      merge      merging the adapter into the base and projecting must reproduce the
                 adapted output. Sequential fine-tuning depends on this identity.

    Usage:
      slm_lora_adapter_check_ex
      slm_lora_adapter_check_ex --in 64 --out 48 --rank 8 --rows 16 --alpha 16
      slm_lora_adapter_check_ex --method dora --seed 3 --verbose
!*/

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/dnn.h>

using namespace std;
using namespace dlib;

/* Reference implementation, in double precision and written from the definition rather
   than from the optimized identities: the merged weight is formed explicitly, its column
   norms are taken directly. It is deliberately the slow, obvious version, since its only
   job is to disagree with the fast one when the fast one is wrong. */
struct reference_adapter
{
    long in_dim = 0, out_dim = 0, rank = 0;
    bool dora = false;
    double s = 0.0;
    vector<double> W, A, B, M;

    void merged_weight(vector<double>& out_w) const
    {
        out_w.assign(static_cast<size_t>(in_dim) * out_dim, 0.0);
        for (long i = 0; i < in_dim; ++i)
            for (long j = 0; j < out_dim; ++j)
            {
                double acc = 0.0;
                for (long r = 0; r < rank; ++r)
                    acc += A[static_cast<size_t>(i) * rank + r] * B[static_cast<size_t>(r) * out_dim + j];
                out_w[static_cast<size_t>(i) * out_dim + j] = W[static_cast<size_t>(i) * out_dim + j] + s * acc;
            }

        if (!dora) return;
        for (long j = 0; j < out_dim; ++j)
        {
            double n2 = 0.0;
            for (long i = 0; i < in_dim; ++i)
            {
                const double v = out_w[static_cast<size_t>(i) * out_dim + j];
                n2 += v * v;
            }
            const double f = M[static_cast<size_t>(j)] / std::sqrt(n2);
            for (long i = 0; i < in_dim; ++i) out_w[static_cast<size_t>(i) * out_dim + j] *= f;
        }
    }

    void forward(const vector<double>& x, long rows, vector<double>& y) const
    {
        vector<double> w;
        merged_weight(w);
        y.assign(static_cast<size_t>(rows) * out_dim, 0.0);
        for (long p = 0; p < rows; ++p)
            for (long j = 0; j < out_dim; ++j)
            {
                double acc = 0.0;
                for (long i = 0; i < in_dim; ++i)
                    acc += x[static_cast<size_t>(p) * in_dim + i] * w[static_cast<size_t>(i) * out_dim + j];
                y[static_cast<size_t>(p) * out_dim + j] = acc;
            }
    }

    // Scalar objective whose gradient the finite differences approximate.
    double loss(const vector<double>& x, long rows, const vector<double>& dy) const
    {
        vector<double> y;
        forward(x, rows, y);
        double acc = 0.0;
        for (size_t i = 0; i < y.size(); ++i) acc += y[i] * dy[i];
        return acc;
    }
};

static void copy_from(const tensor& t, vector<double>& dst)
{
    dst.assign(t.size(), 0.0);
    const float* p = t.host();
    for (size_t i = 0; i < t.size(); ++i) dst[i] = p[i];
}

/* Largest relative discrepancy between an analytic gradient and its finite-difference
   estimate, scaled by the magnitude of the gradient rather than element by element: a
   near-zero component would otherwise dominate the report without meaning anything. */
static double compare(const float* analytic, const vector<double>& numeric)
{
    double scale = 0.0;
    for (size_t i = 0; i < numeric.size(); ++i) scale = std::max(scale, std::abs(numeric[i]));
    if (scale < 1e-12) scale = 1e-12;
    double worst = 0.0;
    for (size_t i = 0; i < numeric.size(); ++i)
        worst = std::max(worst, std::abs(static_cast<double>(analytic[i]) - numeric[i]) / scale);
    return worst;
}

static vector<double> finite_differences(reference_adapter& ref, vector<double>& target,
    const vector<double>& x, long rows, const vector<double>& dy, double eps)
{
    vector<double> g(target.size(), 0.0);
    for (size_t i = 0; i < target.size(); ++i)
    {
        const double keep = target[i];
        target[i] = keep + eps;
        const double lp = ref.loss(x, rows, dy);
        target[i] = keep - eps;
        const double lm = ref.loss(x, rows, dy);
        target[i] = keep;
        g[i] = (lp - lm) / (2.0 * eps);
    }
    return g;
}

static bool run_case(adapter_method method, long in_dim, long out_dim, long rank, long rows,
    double alpha, unsigned long seed, double tol, bool verbose)
{
    const bool dora = (method == adapter_method::dora);
    cout << "\n=== " << adapter_method_name(method)
         << "  (in " << in_dim << ", out " << out_dim << ", rank " << rank
         << ", rows " << rows << ", alpha " << alpha << ")\n";

    dlib::rand rnd(seed);

    resizable_tensor base_w(in_dim, out_dim), x(rows, in_dim), dy(rows, out_dim);
    for (auto& t : { &base_w, &x, &dy })
    {
        float* p = t->host_write_only();
        for (size_t i = 0; i < t->size(); ++i) p[i] = 0.5f * static_cast<float>(rnd.get_random_gaussian());
    }

    low_rank_adapter adapter;
    adapter.configure(in_dim, out_dim, rank, method, alpha);
    const adapter_geometry& geom = adapter.geometry();

    /* The magnitude block belongs to DoRA alone, so parameter_count() leaves it out for
       LoRA. Rather than branch on the presence of a view, the probe blob always reserves
       it and LoRA simply never writes to it: the layer allocates exactly what the
       geometry asks for, a check program can afford the few unused floats. */
    const long blob_size = static_cast<long>(geom.a_count() + geom.b_count() + out_dim);
    resizable_tensor params(1, blob_size), grads(1, blob_size);
    params = 0;
    grads = 0;

    auto a  = adapter.a_view()(params, geom.a_offset(0));
    auto b  = adapter.b_view()(params, geom.b_offset(0));
    auto m  = adapter.m_view()(params, geom.a_count() + geom.b_count());
    auto da = adapter.a_view()(grads, geom.a_offset(0));
    auto db = adapter.b_view()(grads, geom.b_offset(0));
    auto dm = adapter.m_view()(grads, geom.a_count() + geom.b_count());

    adapter.initialize(base_w, a, b, m, rnd);

    bool ok = true;
    auto report = [&](const string& name, double err, double limit, bool exact)
    {
        const bool pass = exact ? (err == 0.0) : (err <= limit);
        cout << "  " << left << setw(22) << name << (pass ? "[ ok ] " : "[FAIL] ")
             << (exact ? "max difference " : "max relative error ")
             << scientific << setprecision(3) << err << defaultfloat << "\n";
        if (!pass) ok = false;
    };

    // --- inertness -------------------------------------------------------------------
    resizable_tensor y(rows, out_dim), y_base(rows, out_dim);
    tt::gemm(0.0f, y_base, 1.0f, x, false, base_w, false);
    memcpy(y, y_base);
    adapter.forward(x, base_w, a, b, m, y);
    {
        double worst = 0.0;
        const float* p1 = y.host();
        const float* p2 = y_base.host();
        for (size_t i = 0; i < y.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(p1[i]) - p2[i]));
        report("inertness", worst, 0.0, true);
    }

    /* Give the adapter something to say. With B still at zero the low-rank path is
       silent and every gradient through it is trivially zero, so the checks below would
       pass on an implementation that does nothing at all. */
    {
        float* pb = b.host();
        for (size_t i = 0; i < b.size(); ++i) pb[i] = 0.4f * static_cast<float>(rnd.get_random_gaussian());
        if (dora)
        {
            float* pm = m.host();
            for (size_t i = 0; i < m.size(); ++i) pm[i] *= 1.0f + 0.3f * static_cast<float>(rnd.get_random_gaussian());
        }
    }

    reference_adapter ref;
    ref.in_dim = in_dim; ref.out_dim = out_dim; ref.rank = rank;
    ref.dora = dora; ref.s = adapter.scale();
    copy_from(base_w, ref.W);
    copy_from(a, ref.A);
    copy_from(b, ref.B);
    if (dora) copy_from(m, ref.M);

    vector<double> xd, dyd;
    copy_from(x, xd);
    copy_from(dy, dyd);

    // --- forward ---------------------------------------------------------------------
    memcpy(y, y_base);
    adapter.forward(x, base_w, a, b, m, y);
    {
        vector<double> y_ref;
        ref.forward(xd, rows, y_ref);
        report("forward vs merged", compare(y.host(), y_ref), tol, false);
    }

    // --- gradients -------------------------------------------------------------------
    resizable_tensor dx(rows, in_dim);
    dx = 0;
    grads = 0;
    adapter.backward(x, base_w, a, b, m, dy, dx, da, db, dm);

    const double eps = 1e-5;
    report("gradient dA", compare(da.host(), finite_differences(ref, ref.A, xd, rows, dyd, eps)), tol, false);
    report("gradient dB", compare(db.host(), finite_differences(ref, ref.B, xd, rows, dyd, eps)), tol, false);
    if (dora)
        report("gradient dm", compare(dm.host(), finite_differences(ref, ref.M, xd, rows, dyd, eps)), tol, false);
    {
        /* dx from the adapter is the low-rank path only; the reference differentiates the
           whole projection, so the frozen base contribution is added before comparing. */
        vector<double> gx = finite_differences(ref, xd, xd, rows, dyd, eps);
        resizable_tensor dx_total(rows, in_dim);
        memcpy(dx_total, dx);
        tt::gemm(1.0f, dx_total, 1.0f, dy, false, base_w, true);
        report("gradient dx", compare(dx_total.host(), gx), tol, false);
    }

    // --- merge -----------------------------------------------------------------------
    {
        resizable_tensor merged(in_dim, out_dim), y_merged(rows, out_dim);
        memcpy(merged, base_w);
        adapter.merge_into_base(a, b, m, merged);
        tt::gemm(0.0f, y_merged, 1.0f, x, false, merged, false);

        double scale = 0.0, worst = 0.0;
        const float* p1 = y_merged.host();
        const float* p2 = y.host();
        for (size_t i = 0; i < y.size(); ++i) scale = std::max(scale, std::abs(static_cast<double>(p2[i])));
        if (scale < 1e-12) scale = 1e-12;
        for (size_t i = 0; i < y.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(p1[i]) - p2[i]) / scale);
        report("merge then project", worst, tol, false);
    }

    if (verbose)
    {
        cout << "  parameters             " << adapter.parameter_count()
             << " floats (A " << geom.a_count() << ", B " << geom.b_count()
             << ", m " << geom.m_count() << ")\n"
             << "  effective scale        " << adapter.scale()
             << "  (alpha " << adapter.alpha() << " / rank " << rank << ")\n"
             << "  adapted fraction       " << fixed << setprecision(2)
             << (100.0 * adapter.parameter_count() / (in_dim * out_dim))
             << "% of the frozen projection" << defaultfloat << "\n";
    }
    return ok;
}

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("in", "Input dimension of the adapted projection (default: 24)", 1);
        parser.add_option("out", "Output dimension of the adapted projection (default: 18)", 1);
        parser.add_option("rank", "Adapter rank (default: 4)", 1);
        parser.add_option("rows", "Number of rows in the probe batch (default: 6)", 1);
        parser.add_option("alpha", "Adapter alpha; the effective scale is alpha / rank (default: 8)", 1);
        parser.add_option("method", "Method to check: lora, dora or both (default: both)", 1);
        parser.add_option("seed", "Random seed (default: 1)", 1);
        parser.add_option("tolerance", "Largest accepted relative error (default: 1e-3)", 1);
        parser.add_option("verbose", "Report the adapter sizing alongside the checks");
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);

        if (parser.option("h"))
        {
            cout << "Numerical validation of the low-rank adaptation core.\n\n";
            parser.print_options();
            return 0;
        }

        auto number = [&](const char* name, long fallback) {
            return parser.option(name) ? std::stol(parser.option(name).argument()) : fallback;
        };
        const long in_dim = number("in", 24);
        const long out_dim = number("out", 18);
        const long rank = number("rank", 4);
        const long rows = number("rows", 6);
        const double alpha = parser.option("alpha")
            ? std::stod(parser.option("alpha").argument()) : 8.0;
        const unsigned long seed = static_cast<unsigned long>(number("seed", 1));
        const double tol = parser.option("tolerance")
            ? std::stod(parser.option("tolerance").argument()) : 1e-3;
        const bool verbose = parser.option("verbose");
        const string which = parser.option("method")
            ? parser.option("method").argument() : string("both");

        /* The finite differences run in double precision on a reference that recomputes
           the merged weight for every perturbed coordinate, so the cost grows with the
           product of the dimensions. Small shapes are the point: a wrong gradient is
           wrong at every size, and here the whole sweep takes seconds. */
        cout << "Checking the low-rank adaptation core against a double-precision reference.\n"
             << "Analytic gradients come from float32 tensors, so agreement is expected "
                "near 1e-4;\na discrepancy of order one is what a real defect looks like.\n";

        bool ok = true;
        if (which == "lora" || which == "both")
            ok &= run_case(adapter_method::lora, in_dim, out_dim, rank, rows, alpha, seed, tol, verbose);
        if (which == "dora" || which == "both")
            ok &= run_case(adapter_method::dora, in_dim, out_dim, rank, rows, alpha, seed, tol, verbose);
        if (which != "lora" && which != "dora" && which != "both")
        {
            cerr << "Unknown method '" << which << "'; expected lora, dora or both.\n";
            return 1;
        }

        cout << "\n" << (ok ? "All checks passed." : "At least one check failed.") << "\n";
        return ok ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        cerr << "Exception thrown: " << e.what() << "\n";
        return 1;
    }
}
