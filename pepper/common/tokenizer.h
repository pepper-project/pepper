#ifndef CODE_PEPPER_COMMON_CIRCUIT_TOKENIZER_H_
#define CODE_PEPPER_COMMON_CIRCUIT_TOKENIZER_H_

#include <istream>

class Tokenizer
{
private:
  std::streampos lastPos;
  std::istream& in;

public:
  Tokenizer(std::istream& is)
    : in(is)
  {
    lastPos = in.tellg();
  }

  bool hasNext()
  {
    return !in.eof();
  }

  std::istream& stream() { return in; }

  template<typename T>
  bool next(T& var)
  {
    bool success = false;
    lastPos = in.tellg();

    if (hasNext())
    {
      in >> var;
      success = in.good();
    }

    return success;
  }

  template<typename T>
  bool peek(T& var)
  {
    bool success = next(var);
    if (success) rewind();
    return success;
  }

  template<typename T>
  T peek()
  {
    T var;
    peek(var);
    return var;
  }

  bool operator >>(int& integer)
  {
    return next(integer);
  }

  bool operator >>(std::string& token)
  {
    return next(token);
  }

  void rewind()
  {
    if (lastPos >= 0)
      in.seekg(lastPos);
  }

  void ignoreLine()
  {
    lastPos = in.tellg();
    std::string str;
    std::getline(in, str);
  }
};

#endif

