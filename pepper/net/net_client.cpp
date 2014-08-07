#include <iostream>

#include <common/utility.h>

#include "net_stat.h"
#include "net_client.h"

using namespace std;

static const int DEFAULT_MSGBUF_LEN = 1024;

NetClient::
NetClient()
  : stat(), msgStat(), msgValid(false),
    msgBufLen(DEFAULT_MSGBUF_LEN), msgHeader("")
{
  msgBuf = new char[msgBufLen];
}

NetClient::
NetClient(const NetClient& other)
  : stat(other.stat), msgStat(other.msgStat), msgValid(other.msgValid),
    msgBufLen(other.msgBufLen), msgHeader(other.msgHeader)
{
  msgBuf = new char[msgBufLen];
  memcpy(msgBuf, other.msgBuf, msgBufLen * sizeof(char));
}

NetClient::
~NetClient()
{
  delete[] msgBuf;
}

NetClient& NetClient::
operator=(const NetClient& other)
{
  if (this != &other)
  {
    stat = other.stat;
    msgStat = other.msgStat;
    msgValid = other.msgValid;
    msgBufLen = other.msgBufLen;
    msgHeader = other.msgHeader;

    delete[] msgBuf;

    msgBuf = new char[msgBufLen];
    memcpy(msgBuf, other.msgBuf, msgBufLen * sizeof(char));
  }
  return *this;
}

NetStat NetClient::
getGlobalStat() const
{
  return stat;
}

NetStat NetClient::
getMessageStat() const
{
  return msgStat;
}

void NetClient::
sendEmpty(const std::string& header)
{
  sendData(header, NULL, 0);
}

void NetClient::
sendData(const string& header, const char* body, size_t len)
{
  send(header, body, len, msgStat);
  stat += msgStat;
}

void NetClient::
sendString(const std::string& header, const std::string& body)
{
  sendData(header, body.c_str(), body.size());
}

void NetClient::
sendFile(const string& header, const string& filename)
{
  FILE* f;
  size_t size = get_file_size(filename.c_str());

  // Still send the file.
  if (!(f = fopen(filename.c_str(), "r")))
    cerr << "[NetClient] Sending non-existant file: " << filename;

  sendFile(header, f, size);

  if (f)
    fclose(f);
}

void NetClient::
sendFile(const string& header, FILE* f, size_t size)
{
  char* buf;

  if (f)
  {
    buf = new char[size];
    size = fread(buf, 1, size, f);
  }
  else
  {
    buf = new char[0];
    size = 0;
  }

  sendData(header, buf, size);
  delete[] buf;
}

template<class T> /*static*/ size_t
outRaw(FILE* f, const T elem);

template<> /*static*/ size_t
outRaw(FILE* f, const mpz_t elem)
{
  return mpz_out_raw(f, elem);
}

template<> /*static*/ size_t
outRaw(FILE* f, const mpq_t elem)
{
  size_t outLen = outRaw(f, mpq_numref(elem));
  if (outLen)
  {
    if ((outLen += outRaw(f, mpq_denref(elem))))
      return outLen;
    else
      return 0;
  }
  return 0;
}

template<class T>
static void
sendVector(NetClient& net, const string& vecname, const T* vec, size_t len)
{
  FILE* f = tmpfile();

  size_t fsize = 0; 
  for (size_t i = 0; i < len; i++)
  {
    size_t outLen = 0;

    if ((outLen = outRaw(f, vec[i])))
      fsize += outLen;
    else
      cerr << "[NetClient] Failed writing a gmp element.";
  }

  rewind(f);
  net.sendFile(vecname, f, fsize);

  fclose(f);
}

void NetClient::
sendZVector(const string& vecname, const mpz_t* vec, size_t len)
{
  sendVector(*this, vecname, vec, len);
}

void NetClient::
sendZVector(const string& vecname, const MPZVector& vec)
{
  sendZVector(vecname, vec.raw_vec(), vec.size());
}

void NetClient::
sendQVector(const string& vecname, const mpq_t* vec, size_t len)
{
  sendVector(*this, vecname, vec, len);
}

void NetClient::
sendQVector(const string& vecname, const MPQVector& vec)
{
  sendQVector(vecname, vec.raw_vec(), vec.size());
}

void NetClient::
reserveMessageBuffer(size_t reqCapacity)
{
  if (msgBufLen > reqCapacity)
    return;

  int newLen = ceil((double) reqCapacity * 1.5);
  delete[] msgBuf;

  msgBuf = new char[newLen];
  msgBufLen = newLen;
}

string NetClient::
waitForMessage()
{
  char* newBuf = recv(msgHeader, msgBuf, msgBufLen * sizeof(char), msgStat);
  stat += msgStat;
  if (newBuf != msgBuf)
  {
    int newBufLen = msgStat.getDataBytesRecv();
    reserveMessageBuffer(newBufLen);
    memcpy(msgBuf, newBuf, newBufLen);
    delete[] newBuf;
  }
  msgValid = true;

  return msgHeader;
}

bool NetClient::
isMessageValid()
{
  return msgValid;
}

void NetClient::
invalidateMessage()
{
  msgValid = false;
}

size_t NetClient::
getData(char* buf, size_t len)
{
  size_t size = min(len * sizeof(char), msgStat.getDataBytesRecv());
  memcpy(buf, msgBuf, size);
  return size;
}

size_t NetClient::
getData(char** buf)
{
  size_t size = msgStat.getDataBytesRecv();
  if (buf)
  {
    char* newBuf = new char[size];
    memcpy(newBuf, msgBuf, size);
    *buf = newBuf;
  }
  return size;
}

size_t NetClient::
getDataAsString(string& msgBody)
{
  size_t strSize = msgStat.getDataBytesRecv();
  msgBody.assign(msgBuf, strSize);
  return strSize;
}

size_t NetClient::
getDataAsFile(const string& filename)
{
  FILE* f;
  size_t size;

  if (!(f = fopen(filename.c_str(), "w")))
  {
    cerr << "[NetClient] File '" << filename << "' doesn't exist.";
    size = 0;
  }
  else
  {
    size = getDataAsFile(f);
    fclose(f);
  }

  return size;
}

size_t NetClient::
getDataAsFile(FILE* f)
{
  size_t size = msgStat.getDataBytesRecv();
  if (f)
    fwrite(msgBuf, 1, size, f);

  return size;
}

template<class T> /*static*/ bool
inRaw(T elem, FILE* f);

template<> /*static*/ bool
inRaw(mpz_t elem, FILE* f)
{
  return mpz_inp_raw(elem, f) != 0;
}

template<> /*static*/ bool
inRaw(mpq_t elem, FILE* f)
{
  if (inRaw(mpq_numref(elem), f))
  {
    if (inRaw(mpq_denref(elem), f))
      return true;
    else
      return false;
  }
  return false;
}

template<class T>
static size_t
getDataAsVector(NetClient& net, T* vec, size_t len)
{
  FILE* f = tmpfile();

  net.getDataAsFile(f);
  rewind(f);

  size_t nvalid;
  for (nvalid = 0; nvalid < len; nvalid++)
  {
    if (!inRaw(vec[nvalid], f))
      break;
  }
  fclose(f);
  return nvalid;
}

size_t NetClient::
getDataAsZVector(mpz_t* vec, size_t len)
{
  return getDataAsVector(*this, vec, len);
}

size_t NetClient::
getDataAsZVector(MPZVector& vec)
{
  return getDataAsZVector(vec.raw_vec(), vec.size());
}

size_t NetClient::
getDataAsQVector(mpq_t* vec, size_t len)
{
  return getDataAsVector(*this, vec, len);
}

size_t NetClient::
getDataAsQVector(MPQVector& vec)
{
  return getDataAsQVector(vec.raw_vec(), vec.size());
}

