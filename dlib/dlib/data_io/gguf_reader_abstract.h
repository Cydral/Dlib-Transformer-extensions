// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GGUF_READER_ABSTRACT_H_
#ifdef DLIB_GGUF_READER_ABSTRACT_H_

#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace dlib
{
    // ---------------------------------------------------------------------------------

    enum class ggml_type : uint32_t
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                The data type of a tensor stored in a GGUF file. The numeric values
                match the ggml enumeration used by the open-weight model ecosystem,
                so the on-disk 32-bit type field can be cast to this enum directly.

            VALUES
                F32, F16                      - Unquantized 32-bit and 16-bit floats
                Q4_0, Q4_1, Q5_0, Q5_1,
                Q8_0, Q8_1                    - Legacy block quantization, 32 values per
                                                block; "_0" forms are symmetric (scale
                                                only), "_1" forms are asymmetric
                                                (scale and minimum)
                Q2_K .. Q8_K                  - K-quant super-block quantization, 256
                                                values per super-block with 6-bit
                                                sub-block scales
                IQ4_NL                        - 4-bit quantization over a fixed
                                                non-linear value table, 32 values per
                                                block; used by converters when a tensor
                                                row is not a multiple of the k-quant
                                                super-block size

            NOTES
                - Quantization mixes such as Q4_K_M are file-level recipes, not tensor
                  types: every tensor in such a file carries one of the types above.
                - Grid-based i-quants (IQ1/IQ2/IQ3 and IQ4_XS) are not listed and are
                  rejected by get_ggml_type_traits().
        !*/
    };

    // ---------------------------------------------------------------------------------

    enum class gguf_type : uint32_t
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                The type of a GGUF metadata value. The numeric values match the GGUF
                specification, so the on-disk 32-bit type field can be cast directly.

            VALUES
                UINT8, INT8, UINT16, INT16,
                UINT32, INT32, UINT64, INT64  - Integer scalars
                FLOAT32, FLOAT64              - Floating-point scalars
                BOOL                          - Boolean scalar (stored as one byte)
                STRING                        - Length-prefixed UTF-8 string
                ARRAY                         - Homogeneous array of any scalar type
        !*/
    };

    // ---------------------------------------------------------------------------------

    struct ggml_type_traits
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The block geometry of a ggml tensor type: how many logical elements
                one quantization block holds and how many bytes that block occupies
                on disk. For unquantized types the block size is 1 and type_size is
                the element size in bytes.

            FIELDS
                block_size - Number of logical elements per block
                type_size  - Number of bytes per block on disk

            NOTES
                The on-disk byte size of a tensor is
                    n_elements / block_size * type_size
                where the first (contiguous) dimension is guaranteed block-aligned
                by the GGUF format.
        !*/

        uint32_t block_size;
        uint32_t type_size;
    };

    ggml_type_traits get_ggml_type_traits(
        ggml_type t
    );
    /*!
        ensures
            - Returns the block geometry (elements per block, bytes per block) of the
              given ggml tensor type.
        throws
            - std::runtime_error if t is not one of the supported types listed in the
              ggml_type documentation (in particular the grid-based i-quants).
    !*/

    // ---------------------------------------------------------------------------------

    struct gguf_value
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A parsed GGUF metadata value. Only the field matching `type` (or
                `array_type` for arrays) is populated. To keep the interface small,
                all integers and booleans are widened to int64_t and all floats to
                double during parsing.

            FIELDS
                type       - The declared GGUF type of the value
                array_type - The element type when type == gguf_type::ARRAY
                i          - Any integer or boolean scalar
                d          - Any floating-point scalar
                s          - String scalar
                arr_str    - Elements of a STRING array
                arr_int    - Elements of an integer or boolean array
                arr_float  - Elements of a FLOAT32/FLOAT64 array
        !*/

        gguf_type type;
        gguf_type array_type;

        int64_t     i;
        double      d;
        std::string s;

        std::vector<std::string> arr_str;
        std::vector<int64_t>     arr_int;
        std::vector<double>      arr_float;

        bool is_array(
        ) const;
        /*!
            ensures
                - Returns true if and only if type == gguf_type::ARRAY.
        !*/

        int64_t as_int(
        ) const;
        /*!
            ensures
                - Returns the scalar as an integer.
                - Floating-point scalars are truncated toward zero.
        !*/

        double as_double(
        ) const;
        /*!
            ensures
                - Returns the scalar as a double.
                - Integer and boolean scalars are converted exactly.
        !*/

        const std::string& as_string(
        ) const;
        /*!
            ensures
                - Returns the string scalar (empty if the value is not a string).
        !*/
    };

    // ---------------------------------------------------------------------------------

    struct gguf_tensor_info
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The descriptor of one tensor inside a GGUF file: its name, data type,
                dimensions, and location inside the tensor-data section. It describes
                the tensor without holding its data; the bytes are read lazily with
                gguf_reader::read_tensor_raw().

            FIELDS
                name   - Tensor name (e.g. "blk.0.attn_q.weight")
                type   - ggml data type of the stored bytes
                dims   - Dimensions in ggml order: dims[0] is the contiguous
                         dimension. A 2D weight of logical shape [out, in] is
                         therefore stored with dims[0] == in and dims[1] == out.
                offset - Byte offset of the tensor data, relative to the start of the
                         aligned tensor-data section (not to the start of the file)
                nbytes - Size of the tensor data on disk, computed from the type's
                         block geometry
        !*/

        std::string name;
        ggml_type type;
        std::vector<uint64_t> dims;
        uint64_t offset;
        uint64_t nbytes;

        uint64_t n_elements(
        ) const;
        /*!
            ensures
                - Returns the product of all dimensions, i.e. the number of logical
                  elements of the tensor.
        !*/

        bool is_quantized(
        ) const;
        /*!
            ensures
                - Returns true if and only if the tensor type is neither F32 nor F16.
        !*/
    };

    // ---------------------------------------------------------------------------------

    class gguf_reader
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A read-only parser for the GGUF container format used to distribute
                open-weight language models. On construction it parses:
                  - the header (magic, version, tensor count, metadata count),
                  - every metadata key/value pair,
                  - every tensor descriptor (name, type, dimensions, data offset).
                The tensor data blob itself is not loaded: individual tensors are
                read on demand with read_tensor_raw(), so arbitrarily large model
                files can be inspected with a small memory footprint.

                This class returns raw, possibly quantized tensor bytes. Conversion
                to float is the responsibility of gguf_dequantize.h, and repacking
                into Dlib network layouts is the responsibility of
                gguf_weight_loader.h.

            SUPPORTED FILES
                - GGUF versions 2 and 3.
                - A little-endian host is assumed, which matches the GGUF on-disk
                  byte order (this is the case on x86/x64 with MSVC and GCC).

            THREAD SAFETY
                All const accessors are safe to call concurrently. read_tensor_raw()
                seeks the underlying stream and must not be called from several
                threads at once on the same object.
        !*/

    public:

        explicit gguf_reader(
            const std::string& path
        );
        /*!
            ensures
                - Opens the file and parses its header, all metadata key/value pairs,
                  and all tensor descriptors.
                - #version() returns the GGUF container version of the file.
                - #metadata() and #tensors() are fully populated.
                - The tensor data section is located (honoring the general.alignment
                  metadata key, defaulting to 32 bytes) but not read.
            throws
                - std::runtime_error if the file cannot be opened, is not a GGUF file,
                  uses an unsupported container version, is truncated, or contains a
                  tensor whose element count is not aligned to its type's block size.
        !*/

        uint32_t version(
        ) const;
        /*!
            ensures
                - Returns the GGUF container version (2 or 3).
        !*/

        const std::map<std::string, gguf_value>& metadata(
        ) const;
        /*!
            ensures
                - Returns all metadata key/value pairs of the file.
        !*/

        const std::vector<gguf_tensor_info>& tensors(
        ) const;
        /*!
            ensures
                - Returns the descriptors of all tensors, in file order.
        !*/

        bool has(
            const std::string& key
        ) const;
        /*!
            ensures
                - Returns true if and only if the metadata contains the given key.
        !*/

        const gguf_value& at(
            const std::string& key
        ) const;
        /*!
            ensures
                - Returns the metadata value stored under the given key.
            throws
                - std::runtime_error if the key is absent.
        !*/

        std::string get_str(
            const std::string& key,
            const std::string& def = ""
        ) const;
        /*!
            ensures
                - Returns at(key).as_string() if the key is present, def otherwise.
        !*/

        int64_t get_int(
            const std::string& key,
            int64_t def = 0
        ) const;
        /*!
            ensures
                - Returns at(key).as_int() if the key is present, def otherwise.
        !*/

        double get_double(
            const std::string& key,
            double def = 0.0
        ) const;
        /*!
            ensures
                - Returns at(key).as_double() if the key is present, def otherwise.
        !*/

        const gguf_tensor_info* find_tensor(
            const std::string& name
        ) const;
        /*!
            ensures
                - Returns a pointer to the descriptor of the tensor with the given
                  name, or nullptr if no such tensor exists.
                - The pointer remains valid for the lifetime of this object.
            notes
                - The lookup is a linear scan over the tensor table. GGUF files hold
                  a few hundred tensors at most, so this is not a bottleneck for the
                  import and runtime-loading workloads this reader serves.
        !*/

        void read_tensor_raw(
            const gguf_tensor_info& t,
            std::vector<uint8_t>& dst
        );
        /*!
            requires
                - t was obtained from this object (via tensors() or find_tensor())
            ensures
                - #dst.size() == t.nbytes
                - #dst holds the raw on-disk bytes of the tensor, still in the
                  quantized representation described by t.type.
            throws
                - std::runtime_error if the read fails (truncated or corrupt file).
        !*/
    };

    // ---------------------------------------------------------------------------------

}

#endif // DLIB_GGUF_READER_ABSTRACT_H_
