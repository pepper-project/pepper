#ifndef CODE_PEPPER_NET_MPI_CLIENT_H_
#define CODE_PEPPER_NET_MPI_CLIENT_H_

#include <string>
#include <cstring>

#include <common/measurement.h>
#include "net_client.h"

class MPIClient : public NetClient
{
  private:
  int sendToRank;
  int recvFromRank;

  Measurement m;

  public:
  MPIClient(int peerRank);
  MPIClient(int sendTo, int recvFrom);

  protected:
  void  send(const std::string& header, const char* body, size_t len, NetStat& stat);
  char* recv(std::string& header, char* buf, size_t bLen, NetStat& stat);

  private:
  size_t send(const char* buf, size_t len, bool isHeader);

  size_t getRecvDataSize(bool isHeader);
  size_t recvHeader(std::string& header);
  size_t recvData(char** data);
  size_t recv(char* buf, size_t len, char** fullBuf, bool isHeader);
};

#endif

