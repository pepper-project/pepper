
#include "net_stat.h"

using namespace std;

NetStat::
NetStat()
{
  clear();
}

NetStat& NetStat::
operator-=(const NetStat& other)
{
  bytesSent -= other.getBytesSent();
  bytesRecv -= other.getBytesRecv();

  bytesSentOverhead -= other.getOverheadBytesSent();
  bytesRecvOverhead -= other.getOverheadBytesRecv();

  sentTime -= other.getSentTime();
  recvTime -= other.getRecvTime();

  return *this;
}

const NetStat NetStat::
operator-(const NetStat& other) const
{
  NetStat out = *this;
  out -= other;
  return out;
}

NetStat& NetStat::
operator+=(const NetStat& other)
{
  bytesSent += other.getBytesSent();
  bytesRecv += other.getBytesRecv();

  bytesSentOverhead += other.getOverheadBytesSent();
  bytesRecvOverhead += other.getOverheadBytesRecv();

  sentTime += other.getSentTime();
  recvTime += other.getRecvTime();

  return *this;
}

const NetStat NetStat::
operator+(const NetStat& other) const
{
  NetStat out = *this;
  out += other;
  return out;
}

void NetStat::
clear()
{
  bytesSent = 0;
  bytesRecv = 0;

  bytesSentOverhead = 0;
  bytesRecvOverhead = 0;

  sentTime = 0;
  recvTime = 0;
}

void NetStat::
setRecvStats(size_t bytesRecv, size_t overhead, double recvTime)
{
  clear();

  if (overhead > bytesRecv)
    overhead = bytesRecv;

  this->bytesRecv = bytesRecv;
  this->bytesRecvOverhead = overhead;
  this->recvTime = recvTime;
}

void NetStat::
setSendStats(size_t bytesSent, size_t overhead, double sentTime)
{
  clear();

  if (overhead > bytesSent)
    overhead = bytesSent;

  this->bytesSent = bytesSent;
  this->bytesSentOverhead = overhead;
  this->sentTime = sentTime;
}

#define SIMPLE_GETTER(getter_name, getter_var, type)  \
type NetStat::                                        \
getter_name() const                                   \
{                                                     \
  return getter_var;                                  \
}

SIMPLE_GETTER(getBytesSent, bytesSent, size_t);
SIMPLE_GETTER(getBytesRecv, bytesRecv, size_t);
SIMPLE_GETTER(getOverheadBytesSent, bytesSentOverhead, size_t);
SIMPLE_GETTER(getOverheadBytesRecv, bytesRecvOverhead, size_t);
SIMPLE_GETTER(getSentTime, sentTime, double);
SIMPLE_GETTER(getRecvTime, recvTime, double);

size_t NetStat::
getDataBytesSent() const
{
  return getBytesSent() - getOverheadBytesSent();
}

size_t NetStat::
getDataBytesRecv() const
{
  return getBytesRecv() - getOverheadBytesRecv();
}

std::ostream& NetStat::
print(const std::string& prefix, bool includeOverhead, std::ostream& os) const
{
  size_t bytesSent = includeOverhead ? getBytesSent() : getDataBytesSent();
  size_t bytesRecv = includeOverhead ? getBytesRecv() : getDataBytesRecv();

  os << prefix << "bytes_sent " << bytesSent << endl;
  os << prefix << "bytes_rcvd " << bytesRecv << endl;
  os << prefix << "send_time_elapsed " << getSentTime() << endl;
  os << prefix << "rcv_time_elapsed " << getRecvTime() << endl;
  return os;
}

ostream& operator<<(ostream& os, const NetStat& stat)
{
  return stat.print("", os);
}

