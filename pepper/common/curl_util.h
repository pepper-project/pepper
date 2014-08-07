#ifndef CODE_PEPPER_COMMON_CURL_UTIL_H_  
#define CODE_PEPPER_COMMON_CURL_UTIL_H_  
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <common/utility.h>

using namespace std;

#ifndef INTERFACE_MPI
#include <curl/curl.h>
#endif

#include <common/measurement.h>
#define HEADER_DISABLE_EXPECT "Expect:"
                              
class CurlUtil {              
  private:                    
#ifndef INTERFACE_MPI
    CURL *curl;
    CURLM *multi_curl;
    CURLcode res;
    struct curl_httppost *formpost;
    struct curl_httppost *lastptr;
    struct curl_slist *headerlist;
#endif
    Measurement m;

    void clean_up();
    
  public:
    CurlUtil();
    void send_file(char *full_file_name, char *upload_url);
    void recv_file(char *full_file_name, char *download_url);
    int get_file_size(char *file_name);
    void send_file(char *full_file_name, char *upload_url, int *size,
		   double *time);
    void recv_file(char *full_file_name, char *download_url, int *size,
		   double *time);
    void get(char *url);
    void get_nonblocking(char *url);
};

#endif  // CODE_PEPPER_COMMON_CURL_UTIL_H_
