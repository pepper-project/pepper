#ifndef CODE_PEPPER_NET_NET_CLIENT_H_
#define CODE_PEPPER_NET_NET_CLIENT_H_

#include <string>
#include <gmp.h>

#include <common/mpnvector.h>

#include "net_stat.h"

class NetClient
{
  NetStat stat;

  // Local (latest message cache)
  NetStat msgStat;
  bool msgValid;
  size_t msgBufLen;
  char* msgBuf;
  std::string msgHeader;

  protected:
  NetClient();

  public:
  NetClient(const NetClient& other);
  NetClient& operator=(const NetClient& other);
  virtual ~NetClient();

  NetStat getGlobalStat() const;
  NetStat getMessageStat() const;

  // In almost all cases, these methods should be used over the raw send() and
  // recv() methods, as they will keep the internal NetStat object updated.
  void sendEmpty(const std::string& header);
  void sendData(const std::string& header, const char* body, size_t len);
  void sendString(const std::string& header, const std::string& body);
  void sendFile(const std::string& header, const std::string& filename);
  void sendFile(const std::string& header, FILE* f, size_t size);
  void sendZVector(const std::string& vecname, const mpz_t* vec, size_t len);
  void sendZVector(const std::string& vecname, const MPZVector& vec);
  void sendQVector(const std::string& vecname, const mpq_t* vec, size_t len);
  void sendQVector(const std::string& vecname, const MPQVector& vec);

  std::string waitForMessage();
  bool isMessageValid();
  void invalidateMessage();

  std::string getHeader();
  size_t getData(char* buf, size_t len);
  size_t getData(char** buf);
  size_t getDataAsString(std::string& msgBody);
  size_t getDataAsFile(const std::string& filename);
  size_t getDataAsFile(FILE* f);
  size_t getDataAsZVector(mpz_t* vec, size_t len);
  size_t getDataAsZVector(MPZVector& vec);
  size_t getDataAsQVector(mpq_t* vec, size_t len);
  size_t getDataAsQVector(MPQVector& vec);

  protected:
  // The methods to actually send the data. These methods should be implemented
  // appropriately by subclasses. They should also not be called directly.
  // Instead, call the sendData and waitForMessage/getData versions of these
  // methods, as they will keep the stat object updated.
  virtual void send(const std::string& header, const char* body, size_t len, NetStat& stat) = 0;
  virtual char* recv(std::string& header, char* buf, size_t bLen, NetStat& stat) = 0;

  private:
  // Note: this throws away the old buffer.
  void reserveMessageBuffer(size_t reqCapacity);
};

#endif

