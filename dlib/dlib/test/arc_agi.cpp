// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/data_io.h>
#include <dlib/rand.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.arc_agi");

    template <typename F>
    bool throws (F f)
    {
        try { f(); }
        catch (const std::exception&) { return true; }
        return false;
    }

    arc_grid_t make_grid (long rows, long cols, const std::vector<int>& cells)
    {
        arc_grid_t g(rows, cols);
        for (long r = 0; r < rows; ++r)
            for (long c = 0; c < cols; ++c)
                g(r, c) = (unsigned char)cells[(size_t)(r * cols + c)];
        return g;
    }

// ----------------------------------------------------------------------------------------

    void test_symmetries ()
    {
        // The eight symmetries of a 2 by 3 grid as numpy computes them in the reference's
        // numbering: rot90 one to three times, fliplr, flipud, the transpose, and fliplr
        // after one rot90.
        const arc_grid_t g = make_grid(2, 3, {0, 1, 2, 3, 4, 5});
        const arc_grid_t expected[8] = {
            make_grid(2, 3, {0, 1, 2, 3, 4, 5}),
            make_grid(3, 2, {2, 5, 1, 4, 0, 3}),
            make_grid(2, 3, {5, 4, 3, 2, 1, 0}),
            make_grid(3, 2, {3, 0, 4, 1, 5, 2}),
            make_grid(2, 3, {2, 1, 0, 5, 4, 3}),
            make_grid(2, 3, {3, 4, 5, 0, 1, 2}),
            make_grid(3, 2, {0, 3, 1, 4, 2, 5}),
            make_grid(3, 2, {5, 2, 4, 1, 3, 0})
        };
        for (int s = 0; s < 8; ++s)
        {
            const arc_grid_t t = arc_apply_symmetry(g, s);
            DLIB_TEST_MSG(t == expected[s], "symmetry " << s);
            DLIB_TEST(arc_apply_symmetry(t, arc_inverse_symmetry(s)) == g);
        }
        DLIB_TEST(arc_apply_symmetry(arc_grid_t(), 1).size() == 0);
    }

// ----------------------------------------------------------------------------------------

    void test_variants ()
    {
        dlib::rand rnd(7);
        for (int trial = 0; trial < 200; ++trial)
        {
            const long rows = 1 + rnd.get_random_32bit_number() % 30;
            const long cols = 1 + rnd.get_random_32bit_number() % 30;
            arc_grid_t g(rows, cols);
            for (long r = 0; r < rows; ++r)
                for (long c = 0; c < cols; ++c)
                    g(r, c) = (unsigned char)(rnd.get_random_32bit_number() % 10);

            arc_variant v;
            v.symmetry = (int)(rnd.get_random_32bit_number() % 8);
            for (unsigned long c = arc_num_colours - 1; c > 1; --c)
                std::swap(v.colours[c], v.colours[1 + rnd.get_random_32bit_number() % c]);

            const arc_grid_t made = arc_apply_variant(g, v);
            DLIB_TEST(arc_undo_variant(made, v) == g);
            // the colours first, then the symmetry
            arc_grid_t mapped = g;
            for (long r = 0; r < rows; ++r)
                for (long c = 0; c < cols; ++c)
                    mapped(r, c) = v.colours[g(r, c)];
            DLIB_TEST(made == arc_apply_symmetry(mapped, v.symmetry));
        }

        const arc_variant v = arc_variant_of("007bbfb7_t3_0213456789");
        DLIB_TEST(arc_task_of("007bbfb7_t3_0213456789") == "007bbfb7");
        DLIB_TEST(v.symmetry == 3);
        DLIB_TEST(v.colours[0] == 0 && v.colours[1] == 2 && v.colours[2] == 1 && v.colours[9] == 9);
        DLIB_TEST(!v.is_identity());

        DLIB_TEST(arc_task_of("007bbfb7") == "007bbfb7");
        DLIB_TEST(arc_variant_of("007bbfb7").is_identity());
        DLIB_TEST(arc_task_of("extra/ConceptARC/AboveBelow/AboveBelow1_t7_0912546837") ==
                  "extra/ConceptARC/AboveBelow/AboveBelow1");
        DLIB_TEST(arc_variant_of("extra/ConceptARC/AboveBelow/AboveBelow1_t7_0912546837").symmetry == 7);

        // A suffix that records no variant leaves the name to the task.
        DLIB_TEST(arc_task_of("abc_t8_0123456789") == "abc_t8_0123456789");
        DLIB_TEST(arc_task_of("abc_t1_0123456788") == "abc_t1_0123456788");
        DLIB_TEST(arc_task_of("abc_t1_012345678") == "abc_t1_012345678");
        DLIB_TEST(arc_variant_of("abc_t1_0123456788").is_identity());
        DLIB_TEST(arc_task_of("_t1_0123456789") == "_t1_0123456789");
    }

// ----------------------------------------------------------------------------------------

    void test_crop ()
    {
        const long side = 30;
        std::vector<unsigned long> window(1 + side * side, arc_pad_token);   // the identifier first
        auto at = [&](long r, long c) -> unsigned long& { return window[(size_t)(1 + r * side + c)]; };

        // A 3 by 4 grid with its end markers.
        for (long r = 0; r < 3; ++r)
            for (long c = 0; c < 4; ++c)
                at(r, c) = arc_colour_offset + (unsigned long)((r + c) % 10);
        for (long c = 0; c < 4; ++c) at(3, c) = arc_eos_token;
        for (long r = 0; r < 3; ++r) at(r, 4) = arc_eos_token;
        arc_grid_t g = arc_decode_grid(window, side * side, side);
        DLIB_TEST(g.nr() == 3 && g.nc() == 4);
        DLIB_TEST(g(2, 3) == 5);

        // The width a row allows never exceeds the rows above it, and the largest area wins.
        std::fill(window.begin(), window.end(), arc_pad_token);
        const unsigned long row0[] = {2, 3, 4, 5, 6}, row1[] = {7, 8, 9}, row2[] = {10, 11, 2, 3};
        for (long c = 0; c < 5; ++c) at(0, c) = row0[c];
        at(0, 5) = arc_eos_token;
        for (long c = 0; c < 3; ++c) at(1, c) = row1[c];
        for (long c = 0; c < 4; ++c) at(2, c) = row2[c];
        g = arc_decode_grid(window, side * side, side);
        DLIB_TEST(g == make_grid(3, 3, {0, 1, 2, 5, 6, 7, 8, 9, 0}));

        std::fill(window.begin(), window.end(), arc_pad_token);
        for (long c = 0; c < 6; ++c) at(0, c) = 7;
        for (long r = 1; r < 8; ++r) at(r, 0) = 6;
        g = arc_decode_grid(window, side * side, side);
        DLIB_TEST(g.nr() == 8 && g.nc() == 1);

        std::fill(window.begin(), window.end(), arc_eos_token);
        DLIB_TEST(arc_decode_grid(window, side * side, side).size() == 0);

        std::fill(window.begin(), window.end(), arc_colour_offset + 9);
        g = arc_decode_grid(window, side * side, side);
        DLIB_TEST(g.nr() == 30 && g.nc() == 30 && g(29, 29) == 9);
    }

// ----------------------------------------------------------------------------------------

    void write_i32 (std::ofstream& out, std::int32_t v)
    {
        const std::uint32_t u = (std::uint32_t)v;
        const char b[4] = {(char)(u & 0xff), (char)((u >> 8) & 0xff), (char)((u >> 16) & 0xff),
                           (char)((u >> 24) & 0xff)};
        out.write(b, 4);
    }

    /* A split of two puzzles in one group, the second a variant of the first, in a window
       of 3 by 3 cells, written as the builder writes it. With corrupt set, the first cell
       of the first input takes the value given. */
    void write_small_split (const string& bin, const string& names, bool corrupt = false, int value = 0)
    {
        const long side = 3;
        const arc_grid_t question = make_grid(1, 2, {1, 2});
        const arc_grid_t answer = make_grid(2, 1, {3, 4});
        const arc_variant v = arc_variant_of("t_t1_0123456789");
        const arc_grid_t grids[4] = {question, answer, arc_apply_variant(question, v), arc_apply_variant(answer, v)};

        auto encode = [&](const arc_grid_t& g) {
            std::vector<int> cells(side * side, (int)arc_pad_token);
            for (long r = 0; r < g.nr(); ++r)
                for (long c = 0; c < g.nc(); ++c)
                    cells[(size_t)(r * side + c)] = (int)(g(r, c) + arc_colour_offset);
            if (g.nr() < side)
                for (long c = 0; c < g.nc(); ++c) cells[(size_t)(g.nr() * side + c)] = (int)arc_eos_token;
            if (g.nc() < side)
                for (long r = 0; r < g.nr(); ++r) cells[(size_t)(r * side + g.nc())] = (int)arc_eos_token;
            return cells;
        };

        std::ofstream out(bin, std::ios::binary);
        out.write("ARCD", 4);
        for (int v32 : {1, (int)(side * side), 12, 2, 3, 3, 2})
            write_i32(out, v32);
        std::vector<int> first = encode(grids[0]);
        if (corrupt) first[0] = value;
        for (int c : first) write_i32(out, c);
        for (int c : encode(grids[2])) write_i32(out, c);
        for (int c : encode(grids[1])) write_i32(out, c);
        for (int c : encode(grids[3])) write_i32(out, c);
        for (int v32 : {0, 5, 6}) write_i32(out, v32);       // identifiers, the blank first
        for (int v32 : {0, 1, 2}) write_i32(out, v32);       // puzzle index
        for (int v32 : {0, 2}) write_i32(out, v32);          // group index
        out.close();

        if (!names.empty())
        {
            std::ofstream n(names);
            n << "<blank>\nt\nt_t1_0123456789\n";
        }
    }

    void test_dataset_file ()
    {
        const string bin = "arc_agi_test_split.bin", names = "arc_agi_test_split.names";
        write_small_split(bin, names);

        const arc_dataset d = load_arc_dataset(bin);
        DLIB_TEST(d.seq_len == 9 && d.colour_vocab == 12);
        DLIB_TEST(d.num_examples() == 2 && d.num_puzzles() == 3 && d.num_groups() == 1);
        DLIB_TEST(d.inputs.size() == 18 && d.labels.size() == 18);
        DLIB_TEST(d.puzzle_names.size() == 3 && d.puzzle_names[2] == "t_t1_0123456789");
        DLIB_TEST(d.puzzle_of(1) == 2 && d.identifier_of(1) == 6);
        DLIB_TEST(d.first_example(2) == 1 && d.end_example(2) == 2);

        const matrix<int, 0, 1> x = arc_make_input(d, 0, d.identifier_of(0));
        DLIB_TEST(x.size() == 10 && x(0) == 12 + 5 && x(1) == 3 && x(2) == 4 && x(3) == 1);
        const matrix<unsigned long, 0, 1> y = arc_make_label(d, 0);
        DLIB_TEST(y(0) == arc_ignore_label && y(1) == 5 && y(2) == 1 && y(3) == arc_ignore_label);

        // The variant is undone back to the task, from its stored grids alone.
        const arc_variant v = arc_variant_of(d.puzzle_names[2]);
        DLIB_TEST(arc_undo_variant(arc_input_grid(d, 1), v) == arc_input_grid(d, 0));
        DLIB_TEST(arc_undo_variant(arc_label_grid(d, 1), v) == arc_label_grid(d, 0));
        DLIB_TEST(arc_label_grid(d, 0) == make_grid(2, 1, {3, 4}));

        // A names file that does not match the split, and a cell beyond the vocabulary.
        {
            std::ofstream n(names);
            n << "<blank>\nt\n";
        }
        DLIB_TEST(throws([&] { load_arc_dataset(bin); }));
        std::remove(names.c_str());
        DLIB_TEST(load_arc_dataset(bin).puzzle_names.empty());

        write_small_split(bin, "", true, 12);
        DLIB_TEST(throws([&] { load_arc_dataset(bin); }));
        write_small_split(bin, "", true, -3);
        DLIB_TEST(throws([&] { load_arc_dataset(bin); }));
        std::remove(bin.c_str());
    }

// ----------------------------------------------------------------------------------------

    void test_vote ()
    {
        const arc_grid_t right = make_grid(1, 2, {1, 2});
        const arc_grid_t wrong = make_grid(2, 1, {1, 2});
        const arc_grid_t other = make_grid(1, 1, {9});

        // Two votes for the wrong grid and one for the right one: one attempt misses, two
        // attempts find it.
        arc_vote vote;
        vote.add("a", 0, right, wrong, 0.1);
        vote.add("a", 0, right, wrong, 0.1);
        vote.add("a", 0, right, right, 0.9);
        DLIB_TEST(vote.num_tasks() == 1 && vote.num_questions() == 1 && vote.num_answers() == 3);
        DLIB_TEST(vote.solved(1) == 0 && vote.solved(2) == 1);
        std::vector<arc_grid_t> a = vote.attempts("a", 0, 2);
        DLIB_TEST(a.size() == 2 && a[0] == wrong && a[1] == right);

        // Equal votes are ranked by their mean confidence.
        vote.add("b", 0, right, other, 0.2);
        vote.add("b", 0, right, right, 0.7);
        DLIB_TEST(vote.attempts("b", 0, 1)[0] == right);
        // and equal confidence by the order they came in
        vote.add("c", 0, right, other, 0.5);
        vote.add("c", 0, right, right, 0.5);
        DLIB_TEST(vote.attempts("c", 0, 1)[0] == other);

        // A task is solved only when every test input is.
        vote.add("d", 0, right, right, 0.5);
        vote.add("d", 1, wrong, other, 0.5);
        DLIB_TEST(vote.num_tasks() == 4 && vote.num_questions() == 5);
        DLIB_TEST(vote.solved(1) == 1);              // b
        DLIB_TEST(vote.solved(2) == 3);              // a, b and c
        DLIB_TEST(vote.attempts("d", 1, 5).size() == 1);
        DLIB_TEST(vote.attempts("e", 0, 5).empty());

        // Two variants expecting different answers to the same input were not undone as
        // they were made.
        DLIB_TEST(throws([&] { vote.add("a", 0, wrong, right, 0.5); }));

        vote.clear();
        DLIB_TEST(vote.num_tasks() == 0 && vote.solved(1) == 0);
    }

// ----------------------------------------------------------------------------------------

    class test_arc_agi : public tester
    {
    public:
        test_arc_agi (
        ) :
            tester ("test_arc_agi",
                    "Runs tests on the ARC-AGI dataset helpers.")
        {}

        void perform_test (
        )
        {
            print_spinner();
            test_symmetries();
            print_spinner();
            test_variants();
            print_spinner();
            test_crop();
            print_spinner();
            test_dataset_file();
            print_spinner();
            test_vote();
        }
    } a;
}
