#include <mpi.h>

#include <libv/mpi_constants.h>

#include "net_stat.h"
#include "mpi_client.h"
#include <string>
#include <cstring>

using namespace std;

enum MPITag
{
  MPI_HEADER_TAG,
  MPI_BODY_TAG
};

static MPI_Datatype
getDatatype(bool isHeader)
{
  return isHeader ? MPI_CHAR : MPI_BYTE;
}

static MPITag
getMPITag(bool isHeader)
{
  return isHeader ? MPI_HEADER_TAG : MPI_BODY_TAG;
}

MPIClient::
MPIClient(int peerRank)
  : sendToRank(peerRank), recvFromRank(peerRank)
{ }

MPIClient::
MPIClient(int outRank, int inRank)
  : sendToRank(outRank), recvFromRank(inRank)
{ }

void MPIClient::
send(const string& header, const char* body, size_t len, NetStat& stat)
{
  m.reset();
  size_t hSize = send(header.c_str(), header.length(), true);
  size_t dSize = send(body, len, false);

  stat.setSendStats(hSize + dSize, hSize, m.get_papi_elapsed_time());
}

size_t MPIClient::
send(const char* buf, size_t len, bool isHeader)
{
  // Because the first argument of MPI_Send isn't declared const.
  char* buf2 = new char[len];
  memcpy(buf2, buf, len);

  m.begin_with_history();
  MPI_Send(buf2, len,
           getDatatype(isHeader), sendToRank,
           getMPITag(isHeader), MPI_COMM_WORLD);
  m.end();

  delete[] buf2;
  return len;
}

char* MPIClient::
recv(string& header, char* buf, size_t bLen, NetStat& stat)
{
  char* fullBuf;
  stat.clear();

  // Clear first because recv() doesn't do it.
  m.reset();
  size_t hSize = recvHeader(header);
  size_t dSize = recv(buf, bLen, &fullBuf, false);

  stat.setRecvStats(hSize + dSize, hSize, m.get_papi_elapsed_time());
  return fullBuf;
}

size_t MPIClient::
getRecvDataSize(bool isHeader)
{
  int size;
  MPI_Status stat;
  MPI_Datatype type = getDatatype(isHeader);
  MPITag tag = getMPITag(isHeader);

  MPI_Probe(recvFromRank, tag, MPI_COMM_WORLD, &stat);
  MPI_Get_count(&stat, type, &size);

  return size;
}

size_t MPIClient::
recvHeader(std::string& header)
{
  size_t hSize = getRecvDataSize(true);
  char* buf = new char[hSize];
  recv(buf, hSize, NULL, true);

  header.assign(buf, hSize);

  return hSize;
}

size_t MPIClient::
recv(char* buf, size_t len, char** fullBuf, bool isHeader)
{
  MPI_Datatype type = getDatatype(isHeader);
  MPITag tag = getMPITag(isHeader);

  size_t msgSize = getRecvDataSize(isHeader);

  // If our buffer is too small, we need to allocate a new buffer to hold the
  // entire message.
  char* recvBuf = buf;
  if (len < msgSize)
  {
    recvBuf = new char[msgSize];
  }

  m.begin_with_history();
  MPI_Recv(recvBuf, max(len, msgSize),
           type, recvFromRank,
           tag, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  m.end();

  if (fullBuf)
    *fullBuf = recvBuf;

  return msgSize;
}

