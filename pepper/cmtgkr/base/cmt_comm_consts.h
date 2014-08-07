#ifndef CODE_PEPPER_CMTGKR_BASE_CMT_COMM_CONSTS_H_
#define CODE_PEPPER_CMTGKR_BASE_CMT_COMM_CONSTS_H_

#include <string>

class CMTCommConsts
{
  public:
  static const std::string BEGIN_CMT;
  static const std::string BEGIN_LAYER;

  static const std::string END_CMT;
  static const std::string END_LAYER;

  static const std::string CONFIGURE;

  static const std::string REQ_OUTPUTS;
  static const std::string REQ_SC_POLY;
  static const std::string REQ_VS;
  static const std::string REQ_LPOLY;

  static const std::string NOTIFY_Z0;
  static const std::string NOTIFY_LPOLY_EXTRAP_PT;

  static const std::string RESPONSE_SUFFIX;

  static std::string response(std::string req)
  {
    return req + RESPONSE_SUFFIX;
  }
};

#endif

