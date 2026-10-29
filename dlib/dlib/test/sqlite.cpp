// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dlib/sqlite.h>
#include <dlib/matrix.h>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.sqlite");

// ----------------------------------------------------------------------------------------

    struct temp_db_file
    {
        explicit temp_db_file(const std::string& name) : path(name) { remove_all(); }
        ~temp_db_file() { remove_all(); }
        void remove_all()
        {
            std::remove(path.c_str());
            std::remove((path + "-journal").c_str());
            std::remove((path + "-wal").c_str());
            std::remove((path + "-shm").c_str());
        }
        std::string path;
    };

    bool throws_sqlite_error(database& db, const std::string& sql, std::string& message)
    {
        try
        {
            db.exec(sql);
        }
        catch (sqlite_error& e)
        {
            message = e.what();
            return true;
        }
        return false;
    }

// ----------------------------------------------------------------------------------------

    void test_library_and_open()
    {
        print_spinner();
        // The header and the library linked with it must be the same release.
        DLIB_TEST(sqlite3_libversion_number() == SQLITE_VERSION_NUMBER);
        DLIB_TEST(std::string(sqlite3_libversion()) == SQLITE_VERSION);
        DLIB_TEST(sqlite3_threadsafe() != 0);
        dlog << LINFO << "SQLite " << sqlite3_libversion() << " (" << sqlite3_sourceid() << ")";

        database db;
        DLIB_TEST(db.is_open() == false);
        db.open(":memory:");
        DLIB_TEST(db.is_open());
        DLIB_TEST(db.get_database_filename() == ":memory:");
        DLIB_TEST(query_int(db, "select 1+1") == 2);

        // A file that cannot be created must leave the object closed.
        database bad;
        bool thrown = false;
        try
        {
            bad.open("dtest_sqlite_missing_dir/sub/never.db");
        }
        catch (sqlite_error& e)
        {
            thrown = true;
            DLIB_TEST(std::string(e.what()).size() > 0);
        }
        DLIB_TEST(thrown);
        DLIB_TEST(bad.is_open() == false);
        DLIB_TEST(bad.get_database_filename() == "dtest_sqlite_missing_dir/sub/never.db");
    }

// ----------------------------------------------------------------------------------------

    void test_binding_and_reading()
    {
        print_spinner();
        database db(":memory:");
        db.exec("create table t (id integer primary key, i integer, i64 integer, u integer, "
                "d real, f real, s text, b blob, n integer, obj blob)");

        statement ins(db, "insert into t (i, i64, u, d, f, s, b, n, obj) values (?,?,?,?,?,?,?,?,?)");
        DLIB_TEST(ins.get_max_parameter_id() >= 9);

        const int ival = -123456;
        const int64 i64val = std::numeric_limits<int64>::max() - 7;
        const unsigned int uval = std::numeric_limits<unsigned int>::max();
        const double dval = 3.141592653589793;
        const float fval = 2.5f;
        std::string text = "plain";
        text += '\0';
        text += "after nul \xC3\xA9t\xC3\xA9 \xE2\x82\xAC";
        std::vector<char> blob;
        for (int k = 0; k < 300; ++k)
            blob.push_back(static_cast<char>(k));
        std::vector<int> obj = {1, 2, 3, 5, 8, 13};

        ins.bind(1, ival);
        ins.bind(2, i64val);
        ins.bind(3, uval);
        ins.bind(4, dval);
        ins.bind(5, fval);
        ins.bind(6, text);
        ins.bind_blob(7, blob);
        ins.bind_null(8);
        ins.bind(9, obj);
        ins.exec();
        const int64 first_id = db.last_insert_rowid();
        DLIB_TEST(first_id == 1);

        // An empty vector is a BLOB of length zero, not NULL, and an empty string is TEXT.
        ins.bind(1, 0);
        ins.bind(2, int64(0));
        ins.bind(3, 0u);
        ins.bind(4, 0.0);
        ins.bind(5, 0.0f);
        ins.bind(6, std::string());
        ins.bind_blob(7, std::vector<char>());
        ins.bind_null(8);
        ins.bind(9, std::vector<int>());
        ins.exec();
        DLIB_TEST(db.last_insert_rowid() == 2);

        statement sel(db, "select i, i64, u, d, f, s, b, n, obj, typeof(b), octet_length(s) from t where id = ?");
        sel.bind(1, first_id);
        sel.exec();
        DLIB_TEST(sel.move_next());
        DLIB_TEST(sel.get_num_columns() == 11);
        DLIB_TEST(sel.get_column_name(0) == "i");

        int i = 0; int64 i64 = 0; unsigned int u = 0; double d = 0; float f = 0;
        std::string s; std::vector<int> o;
        sel.get_column(0, i);
        sel.get_column(1, i64);
        sel.get_column(2, u);
        sel.get_column(3, d);
        sel.get_column(4, f);
        sel.get_column(5, s);
        sel.get_column(8, o);
        DLIB_TEST(i == ival);
        DLIB_TEST(i64 == i64val);
        DLIB_TEST(u == uval);
        DLIB_TEST(d == dval);
        DLIB_TEST(f == fval);
        DLIB_TEST_MSG(s == text, "text with an embedded NUL came back with " << s.size() << " bytes");
        DLIB_TEST(sel.get_column_as_blob(6) == blob);
        DLIB_TEST(sel.get_column_as_text(7).empty());
        DLIB_TEST(sel.get_column_as_blob(7).empty());
        DLIB_TEST(sel.get_column_as_int(7) == 0);
        DLIB_TEST(o == obj);
        DLIB_TEST(sel.get_column_as_text(9) == "blob");
        DLIB_TEST(sel.get_column_as_int64(10) == static_cast<int64>(text.size()));
        DLIB_TEST(sel.move_next() == false);
        DLIB_TEST(sel.get_num_columns() == 0);

        sel.bind(1, int64(2));
        sel.exec();
        DLIB_TEST(sel.move_next());
        DLIB_TEST(sel.get_column_as_text(9) == "blob");
        DLIB_TEST(sel.get_column_as_blob(6).empty());
        std::vector<int> empty_obj = {42};
        sel.get_column(8, empty_obj);
        DLIB_TEST(empty_obj.empty());
        DLIB_TEST(sel.get_column_as_text(5).empty());
        DLIB_TEST(sel.move_next() == false);

        // Named parameters and statement reuse.
        statement named(db, "select :a * :b");
        const unsigned long pa = named.get_parameter_id(":a");
        const unsigned long pb = named.get_parameter_id(":b");
        DLIB_TEST(pa == 1 && pb == 2);
        DLIB_TEST(named.get_parameter_id(":missing") == 0);
        for (int k = 1; k <= 20; ++k)
        {
            named.bind(pa, k);
            named.bind(pb, k + 1);
            named.exec();
            DLIB_TEST(named.move_next());
            DLIB_TEST(named.get_column_as_int(0) == k * (k + 1));
        }

        // dlib objects stored as BLOBs.
        matrix<double> m = randm(5, 7);
        statement put(db, "insert into t (obj) values (?)");
        put.bind(1, m);
        put.exec();
        matrix<double> back;
        query_object(db, "select obj from t where id = " + std::to_string(db.last_insert_rowid()), back);
        DLIB_TEST(back == m);
    }

// ----------------------------------------------------------------------------------------

    void test_errors()
    {
        print_spinner();
        database db(":memory:");
        db.exec("create table u (k text unique not null)");
        db.exec("insert into u values ('one')");

        std::string message;
        DLIB_TEST(throws_sqlite_error(db, "insert into u values ('one')", message));
        DLIB_TEST_MSG(message.find("UNIQUE") != std::string::npos, message);
        DLIB_TEST(throws_sqlite_error(db, "insert into u values (null)", message));
        DLIB_TEST_MSG(message.find("NOT NULL") != std::string::npos, message);

        bool thrown = false;
        try { statement bad(db, "selec nothing"); }
        catch (sqlite_error& e) { thrown = true; message = e.what(); }
        DLIB_TEST(thrown);
        DLIB_TEST_MSG(message.find("syntax") != std::string::npos, message);

        // Double-quoted strings are identifiers only in the bundled build, a string
        // literal never.
        if (sqlite3_compileoption_used("DQS=0"))
            DLIB_TEST(throws_sqlite_error(db, "select \"not a column\" from u", message));

        thrown = false;
        try { query_int(db, "select k from u union all select 'two'"); }
        catch (sqlite_error&) { thrown = true; }
        DLIB_TEST(thrown);
        thrown = false;
        try { query_text(db, "select k from u where 0"); }
        catch (sqlite_error&) { thrown = true; }
        DLIB_TEST(thrown);
    }

// ----------------------------------------------------------------------------------------

    void test_transactions()
    {
        print_spinner();
        database db(":memory:");
        db.exec("create table v (x integer)");

        {
            transaction trans(db);
            db.exec("insert into v values (1)");
            trans.commit();
            trans.commit();
        }
        DLIB_TEST(query_int(db, "select count(*) from v") == 1);

        {
            transaction trans(db);
            db.exec("insert into v values (2)");
        }
        DLIB_TEST(query_int(db, "select count(*) from v") == 1);

        // An exception leaving the scope rolls the transaction back without terminating.
        try
        {
            transaction trans(db);
            db.exec("insert into v values (3)");
            throw std::runtime_error("leave the scope");
        }
        catch (std::runtime_error&) {}
        DLIB_TEST(query_int(db, "select count(*) from v") == 1);

        // A commit refused by a deferred constraint leaves the transaction open, and the
        // destructor rolls it back, after which the connection accepts new transactions.
        db.exec("pragma foreign_keys = on");
        db.exec("create table parent (id integer primary key)");
        db.exec("create table child (pid integer references parent(id) deferrable initially deferred)");
        bool thrown = false;
        try
        {
            transaction trans(db);
            db.exec("insert into child values (99)");
            trans.commit();
        }
        catch (sqlite_error& e)
        {
            thrown = true;
            DLIB_TEST_MSG(std::string(e.what()).find("FOREIGN KEY") != std::string::npos, e.what());
        }
        DLIB_TEST(thrown);
        DLIB_TEST(query_int(db, "select count(*) from child") == 0);
        {
            transaction trans(db);
            db.exec("insert into parent values (99)");
            db.exec("insert into child values (99)");
            trans.commit();
        }
        DLIB_TEST(query_int(db, "select count(*) from child") == 1);
    }

// ----------------------------------------------------------------------------------------

    void test_query_tools()
    {
        print_spinner();
        database db(":memory:");
        DLIB_TEST(query_int(db, "select 7") == 7);
        DLIB_TEST(query_int64(db, "select 9007199254740993") == 9007199254740993LL);
        DLIB_TEST(std::abs(query_double(db, "select 0.25 * 4") - 1.0) < 1e-12);
        DLIB_TEST(query_text(db, "select 'abc' || 'def'") == "abcdef");
        const std::vector<char> b = query_blob(db, "select x'00ff10'");
        DLIB_TEST(b.size() == 3);
        DLIB_TEST(static_cast<unsigned char>(b[0]) == 0x00);
        DLIB_TEST(static_cast<unsigned char>(b[1]) == 0xff);
        DLIB_TEST(static_cast<unsigned char>(b[2]) == 0x10);
        if (sqlite3_compileoption_used("ENABLE_MATH_FUNCTIONS"))
            DLIB_TEST(std::abs(query_double(db, "select sqrt(2.0) * sqrt(2.0)") - 2.0) < 1e-12);
        DLIB_TEST(query_text(db, "select json_extract('{\"a\":[1,2,{\"b\":\"ok\"}]}', '$.a[2].b')") == "ok");
    }

// ----------------------------------------------------------------------------------------

    void test_bundled_options()
    {
        print_spinner();
        database db(":memory:");
        if (sqlite3_compileoption_used("ENABLE_FTS5"))
        {
            db.exec("create virtual table docs using fts5(title, body)");
            statement ins(db, "insert into docs values (?, ?)");
            ins.bind(1, std::string("Swans"));
            ins.bind(2, std::string("Albireo is the double star in the head of the swan"));
            ins.exec();
            ins.bind(1, std::string("Bees"));
            ins.bind(2, std::string("A hive keeps its honey in wax cells"));
            ins.exec();
            DLIB_TEST(query_text(db, "select title from docs where docs match 'swan'") == "Swans");
            DLIB_TEST(query_int(db, "select count(*) from docs where docs match 'honey OR star'") == 2);
        }
        if (sqlite3_compileoption_used("SECURE_DELETE"))
            DLIB_TEST(query_int(db, "pragma secure_delete") == 1);
        if (sqlite3_compileoption_used("OMIT_LOAD_EXTENSION"))
        {
            std::string message;
            DLIB_TEST(throws_sqlite_error(db, "select load_extension('nothing')", message));
        }
    }

// ----------------------------------------------------------------------------------------

    void test_file_database_and_threads()
    {
        print_spinner();
        temp_db_file file("dtest_sqlite_file.db");
        {
            database db(file.path);
            DLIB_TEST(query_text(db, "pragma journal_mode = wal") == "wal");
            db.exec("create table w (thread integer, k integer)");
        }

        // One connection per thread, the threads writing to the same file at once.
        const int num_threads = 4;
        const int rows_per_thread = 50;
        std::vector<std::thread> threads;
        std::vector<std::string> failures(num_threads);
        for (int t = 0; t < num_threads; ++t)
        {
            threads.emplace_back([&, t]() {
                try
                {
                    database db(file.path);
                    db.exec("pragma busy_timeout = 10000");
                    statement ins(db, "insert into w values (?, ?)");
                    for (int k = 0; k < rows_per_thread; ++k)
                    {
                        transaction trans(db);
                        ins.bind(1, t);
                        ins.bind(2, k);
                        ins.exec();
                        trans.commit();
                    }
                }
                catch (std::exception& e)
                {
                    failures[t] = e.what();
                }
            });
        }
        for (auto& th : threads)
            th.join();
        for (const auto& f : failures)
            DLIB_TEST_MSG(f.empty(), f);

        database again(file.path);
        DLIB_TEST(query_int(again, "select count(*) from w") == num_threads * rows_per_thread);
        DLIB_TEST(query_int(again, "select count(distinct thread) from w") == num_threads);

        // A large text value goes through the binding and comes back whole.
        std::string big(1 << 20, 'x');
        for (size_t k = 0; k < big.size(); k += 997)
            big[k] = static_cast<char>('a' + k % 26);
        again.exec("create table big (v text)");
        statement ins(again, "insert into big values (?)");
        ins.bind(1, big);
        ins.exec();
        DLIB_TEST(query_text(again, "select v from big") == big);
    }

// ----------------------------------------------------------------------------------------

    class sqlite_tester : public tester
    {
    public:
        sqlite_tester (
        ) :
            tester ("test_sqlite",
                    "Runs tests on the sqlite component and the SQLite library it uses.")
        {}

        void perform_test (
        )
        {
            test_library_and_open();
            test_binding_and_reading();
            test_errors();
            test_transactions();
            test_query_tools();
            test_bundled_options();
            test_file_database_and_threads();
        }
    } a;

}
