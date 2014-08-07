#ifndef CODE_PEPPER_CMTGKR_BASE_CMT_PROVER_H_
#define CODE_PEPPER_CMTGKR_BASE_CMT_PROVER_H_

#include <string>

#include <net/net_client.h>
#include "cmt_protocol_config.h"

class CMTProver
{
  protected:
  NetClient& net;

  CMTProverConfig conf;

  public:
  CMTProver(NetClient& netClient);
  CMTProver(NetClient& netClient, CMTProverConfig cc);
};

#endif

