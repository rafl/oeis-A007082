#pragma once

#include <algorithm>
#include <exception>
#include <stdint.h>
#include <vector>
#include <string>
#include <format>

struct Exception : std::exception {
    Exception(std::string const & what) : mWhat(what) {

    }

    const char* what() const noexcept override
    {
        return mWhat.c_str();
    }
    

    std::string mWhat; 
};

#define ASSERT_MSG(x, msg, ...) do { \
     if (!(x)) { throw Exception(std::format("Assertion at {}:{}\nmsg: {}", __FILE__, __LINE__, std::format(msg, ##__VA_ARGS__))); } \
} while(false)

#define ASSERT(x) ASSERT_MSG(x, "");

typedef uint64_t coefficient_t;

coefficient_t mult(coefficient_t a, coefficient_t b)
{
    coefficient_t res = a * b;
    ASSERT_MSG(b == 0 || (res / b) == a, "Multiplication overflowed");
    return res;
}

struct CyclomaticFieldValue {
    CyclomaticFieldValue()
    {
        // Maybe delete in future;
    }

    CyclomaticFieldValue(size_t m) : mCoefficients(m, 0), mDenominator(1)
    {

    }

    static CyclomaticFieldValue One(size_t m)
    {
        CyclomaticFieldValue val(m);
        val.mCoefficients[0] = 1;
        return val;
    }

    static CyclomaticFieldValue Zero(size_t m)
    {
        return CyclomaticFieldValue(m);
    }

    static CyclomaticFieldValue Omega(size_t m)
    {
        CyclomaticFieldValue val(m);
        val.mCoefficients[1] = 1;
        return val;
    }

    void Accumulate(CyclomaticFieldValue const & b)
    {
        ASSERT_MSG(mDenominator == 1 && b.mDenominator == 1, "Denoms wrong");
        ASSERT(mCoefficients.size() == b.mCoefficients.size());
        for (size_t i = 0; i < mCoefficients.size(); i++)
        {
            mCoefficients[i] += b.mCoefficients[i];
        }
    }

    static CyclomaticFieldValue Add(CyclomaticFieldValue const & a, CyclomaticFieldValue const & b)
    {
        CyclomaticFieldValue ret = a;
        ret.Accumulate(b);
        return ret;
    }

    static CyclomaticFieldValue Multiply(CyclomaticFieldValue const & a, CyclomaticFieldValue const & b)
    {
       ASSERT(a.mCoefficients.size() == b.mCoefficients.size());
       auto m = a.mCoefficients.size();

       auto ret = CyclomaticFieldValue::Zero(m);

       for (size_t i = 0; i < a.mCoefficients.size(); i++)
        {
            for (size_t j = 0; i < b.mCoefficients.size(); i++)
            {
                ret.mCoefficients[(i + j) % m] += mult(a.mCoefficients[i], b.mCoefficients[j]); 
            }
        }
        
        ret.mDenominator = mult(a.mDenominator, b.mDenominator);
    }

    void MultiplyBy(CyclomaticFieldValue const & b)
    {
        *this = CyclomaticFieldValue::Multiply(*this, b);
    }

    /* To work out y = 1/x - instead solve x * y = 1
    
    */
    static CyclomaticFieldValue Invert(CyclomaticFieldValue & a)
    {
        (void) a;
        ASSERT("TODO");
    }

    void Decrement(CyclomaticFieldValue const & b)
    {
        ASSERT_MSG(mDenominator == 1 && b.mDenominator == 1, "Denoms wrong");
        ASSERT(mCoefficients.size() == b.mCoefficients.size());
        for (size_t i = 0; i < mCoefficients.size(); i++)
        {
            mCoefficients[i] -= b.mCoefficients[i];
        }
    }

    static CyclomaticFieldValue Subtract(CyclomaticFieldValue const & a, CyclomaticFieldValue const & b)
    {
        CyclomaticFieldValue ret = a;
        ret.Decrement(b);
        return ret;
    }

    bool IsZero()
    {
        return std::ranges::all_of(mCoefficients, [](coefficient_t a){return a == 0;});
    }

    static CyclomaticFieldValue Negate(CyclomaticFieldValue const & a)
    {
        // TODO better
        return CyclomaticFieldValue::Subtract(CyclomaticFieldValue::Zero(a.mCoefficients.size()), a);
    }




    // (a + b*w + c*w^2 +... ) / denom(int) and a, b, c all ints
    // (a/a_denom + b/b_denom*w + c/c_denom*w^2 +... )
    // (a + b*w + c*w^2 +... ) / (a + b*w + c*w^2 +... )

    // [a, b, c, d] -> a + b*w + c*w^2 +...
   std::vector<coefficient_t> mCoefficients;
   coefficient_t mDenominator;
};

typedef CyclomaticFieldValue field_value_t;