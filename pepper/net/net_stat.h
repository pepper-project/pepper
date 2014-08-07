#ifndef CODE_PEPPER_NET_NET_STAT_H_
#define CODE_PEPPER_NET_NET_STAT_H_

#include <iostream>

class NetStat
{
  private:
  size_t bytesSent;
  size_t bytesRecv;

  size_t bytesSentOverhead;
  size_t bytesRecvOverhead;

  double sentTime;
  double recvTime;

  public:
  NetStat();

  NetStat& operator-=(const NetStat& other);
  const NetStat operator-(const NetStat& other) const;

  NetStat& operator+=(const NetStat& other);
  const NetStat operator+(const NetStat& other) const;

  void clear();
  void setRecvStats(size_t bytesRecv, size_t overhead, double recvTime);
  void setSendStats(size_t bytesSent, size_t overhead, double sentTime);

  size_t getBytesSent() const;
  size_t getBytesRecv() const;

  size_t getOverheadBytesSent() const;
  size_t getOverheadBytesRecv() const;

  size_t getDataBytesSent() const;
  size_t getDataBytesRecv() const;

  double getSentTime() const;
  double getRecvTime() const;

  std::ostream& print(const std::string& prefix = "",
                      bool includeOverhead = false,
                      std::ostream& os = std::cout) const;

  friend std::ostream& operator<<(std::ostream& os, const NetStat& stat);
};

#endif

