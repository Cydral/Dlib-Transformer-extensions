// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SQLiTE_TOOLS_ABSTRACT_H_
#ifdef DLIB_SQLiTE_TOOLS_ABSTRACT_H_


#include "sqlite_abstract.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{

    class transaction : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for creating exception safe
                database transactions.
        !*/

    public:
        transaction (
            database& db
        );
        /*!
            ensures
                - Begins a database transaction which will be rolled back
                  if commit() isn't called eventually.
                - In particular, performs: db.exec("begin transaction");
        !*/

        void commit (
        );
        /*!
            ensures
                - if (commit() hasn't already succeeded) then
                    - Commits all changes made during this database transaction.
                    - In particular, performs: db.exec("commit");
                - else
                    - does nothing
            throws
                - sqlite_error
                    If the commit fails, the transaction stays uncommitted and the
                    destructor rolls it back.
        !*/

        ~transaction(
        );
        /*!
            ensures
                - if (commit() never succeeded) then
                    - rolls back any changes made to the database during this transaction.
                    - In particular, performs: db.exec("rollback");
                - else
                    - does nothing
            throws
                - sqlite_error
                    If the rollback fails.  When the destructor runs because another
                    exception propagates, a failed rollback is not reported, since a
                    second exception would terminate the program.
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void query_object (
        database& db,
        const std::string& query,
        T& item
    );
    /*!
        ensures
            - executes the given SQL query against db.  If the query results in a 
              single row and column being returned then the data in the column is 
              interpreted as a binary BLOB and deserialized into item.
        throws
            - sqlite_error or serialization_error if an error occurs which prevents
              this operation from succeeding.

    !*/

// ----------------------------------------------------------------------------------------

    std::string query_text (
        database& db,
        const std::string& query
    );
    /*!
        ensures
            - executes the given SQL query against db.  If the query results in a 
              single row and column being returned then the data in the column is 
              converted to text and returned.
        throws
            - sqlite_error if an error occurs which prevents this operation from 
              succeeding.
    !*/

// ----------------------------------------------------------------------------------------

    double query_double (
        database& db,
        const std::string& query
    );
    /*!
        ensures
            - executes the given SQL query against db.  If the query results in a 
              single row and column being returned then the data in the column is 
              converted to a double and returned.
        throws
            - sqlite_error if an error occurs which prevents this operation from 
              succeeding.
    !*/

// ----------------------------------------------------------------------------------------

    int query_int (
        database& db,
        const std::string& query
    );
    /*!
        ensures
            - executes the given SQL query against db.  If the query results in a 
              single row and column being returned then the data in the column is 
              converted to an int and returned.
        throws
            - sqlite_error if an error occurs which prevents this operation from 
              succeeding.
    !*/

// ----------------------------------------------------------------------------------------

    int64 query_int64 (
        database& db,
        const std::string& query
    );
    /*!
        ensures
            - executes the given SQL query against db.  If the query results in a 
              single row and column being returned then the data in the column is 
              converted to an int64 and returned.
        throws
            - sqlite_error if an error occurs which prevents this operation from 
              succeeding.
    !*/

// ----------------------------------------------------------------------------------------

    const std::vector<char> query_blob (
        database& db,
        const std::string& query
    );
    /*!
        ensures
            - executes the given SQL query against db.  If the query results in a 
              single row and column being returned then the data in the column is 
              returned as a binary BLOB.
        throws
            - sqlite_error if an error occurs which prevents this operation from 
              succeeding.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SQLiTE_TOOLS_H_


