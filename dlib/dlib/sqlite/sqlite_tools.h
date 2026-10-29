// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SQLiTE_TOOLS_H_
#define DLIB_SQLiTE_TOOLS_H_


#include "sqlite_tools_abstract.h"
#include <exception>
#include "sqlite.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{

    namespace impl
    {
        inline bool exception_in_flight (
        )
        {
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
            return std::uncaught_exceptions() > 0;
#else
            return std::uncaught_exception();
#endif
        }
    }

// ----------------------------------------------------------------------------------------

    class transaction : noncopyable
    {
    public:
        transaction (
            database& db_
        ) :
            db(db_),
            committed(false)
        {
            db.exec("begin transaction");
        }

        void commit ()
        {
            if (!committed)
            {
                // Marked as committed only once the commit succeeded: a commit that throws,
                // on a busy database for instance, leaves the transaction open, and the
                // destructor then rolls it back.
                db.exec("commit");
                committed = true;
            }
        }

        ~transaction() noexcept(false)
        {
            if (!committed)
            {
                // A second exception thrown while another one propagates would terminate the
                // program, so a failed rollback is only reported when nothing else is.
                if (impl::exception_in_flight())
                {
                    try { db.exec("rollback"); } catch (...) {}
                }
                else
                {
                    db.exec("rollback");
                }
            }
        }

    private:
        database& db;
        bool committed;

    };

// ----------------------------------------------------------------------------------------


    template <
        typename T
        >
    void query_object (
        database& db,
        const std::string& query,
        T& item
    )
    {
        statement st(db, query);
        st.exec();
        if (st.move_next() && st.get_num_columns() == 1)
        {
            st.get_column_as_object(0,item);
            if (st.move_next())
                throw sqlite_error("query doesn't result in exactly 1 element");
        }
        else
        {
            throw sqlite_error("query doesn't result in exactly 1 element");
        }
    }

// ----------------------------------------------------------------------------------------

    inline std::string query_text (
        database& db,
        const std::string& query
    )
    {
        statement st(db, query);
        st.exec();
        if (st.move_next() && st.get_num_columns() == 1)
        {
            const std::string& temp = st.get_column_as_text(0);
            if (st.move_next())
                throw sqlite_error("query doesn't result in exactly 1 element");
            return temp;
        }
        else
        {
            throw sqlite_error("query doesn't result in exactly 1 element");
        }
    }

// ----------------------------------------------------------------------------------------

    inline double query_double (
        database& db,
        const std::string& query
    )
    {
        statement st(db, query);
        st.exec();
        if (st.move_next() && st.get_num_columns() == 1)
        {
            double temp = st.get_column_as_double(0);
            if (st.move_next())
                throw sqlite_error("query doesn't result in exactly 1 element");
            return temp;
        }
        else
        {
            throw sqlite_error("query doesn't result in exactly 1 element");
        }
    }

// ----------------------------------------------------------------------------------------

    inline int query_int (
        database& db,
        const std::string& query
    )
    {
        statement st(db, query);
        st.exec();
        if (st.move_next() && st.get_num_columns() == 1)
        {
            int temp = st.get_column_as_int(0);
            if (st.move_next())
                throw sqlite_error("query doesn't result in exactly 1 element");
            return temp;
        }
        else
        {
            throw sqlite_error("query doesn't result in exactly 1 element");
        }
    }

// ----------------------------------------------------------------------------------------

    inline int64 query_int64 (
        database& db,
        const std::string& query
    )
    {
        statement st(db, query);
        st.exec();
        if (st.move_next() && st.get_num_columns() == 1)
        {
            int64 temp = st.get_column_as_int64(0);
            if (st.move_next())
                throw sqlite_error("query doesn't result in exactly 1 element");
            return temp;
        }
        else
        {
            throw sqlite_error("query doesn't result in exactly 1 element");
        }
    }

// ----------------------------------------------------------------------------------------

    inline const std::vector<char> query_blob (
        database& db,
        const std::string& query
    )
    {
        statement st(db, query);
        st.exec();
        if (st.move_next() && st.get_num_columns() == 1)
        {
            const std::vector<char>& temp = st.get_column_as_blob(0);
            if (st.move_next())
                throw sqlite_error("query doesn't result in exactly 1 element");
            return temp;
        }
        else
        {
            throw sqlite_error("query doesn't result in exactly 1 element");
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SQLiTE_TOOLS_H_

