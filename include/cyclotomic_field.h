#pragma once

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

    CyclomaticFieldValue MultiplyBy( CyclomaticFieldValue const & b)
    {
        ASSERT(mCoefficients.size() == b.mCoefficients.size());
        for (size_t i = 0; i < mCoefficients.size(); i++)
        {
            mCoefficients[i] = mult(mCoefficients[i], b.mCoefficients[i]);
        }
        
        mDenominator = mult(mDenominator, b.mDenominator);
    }

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

    static CyclomaticFieldValue Subtract(CyclomaticFieldValue & a, CyclomaticFieldValue & b)
    {
        CyclomaticFieldValue ret = a;
        ret.Decrement(b);
        return ret;
    }



    // [a, b, c, d] -> a + b*w + c*w^2 +...
   std::vector<coefficient_t> mCoefficients;
   coefficient_t mDenominator;
};

typedef CyclomaticFieldValue field_value_t;