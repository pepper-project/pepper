#include <common/curl_util.h>

CurlUtil::CurlUtil() {
#ifndef INTERFACE_MPI
  curl = NULL;
  formpost = NULL;
  headerlist = NULL;
  curl_global_init(CURL_GLOBAL_ALL);
  headerlist = curl_slist_append(headerlist, HEADER_DISABLE_EXPECT);
#endif
}

int CurlUtil::get_file_size(char *file_name) {
#ifndef INTERFACE_MPI
  struct stat filest;
  stat(file_name, &filest);
  return filest.st_size;
#else
  return 0;
#endif
}

void CurlUtil::send_file(char *full_file_name, char *upload_url,
                         int *size, double *time) {
#ifndef INTERFACE_MPI
  curl = curl_easy_init();

  // some of this code is from example code in libcurl
  formpost = NULL;
  lastptr = NULL;

  curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "file",
               CURLFORM_FILE, full_file_name, CURLFORM_END);

  curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "submit",
               CURLFORM_COPYCONTENTS, "send", CURLFORM_END);

  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, upload_url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
    curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
    //curl_easy_setopt(curl, CURLOPT_HEADER, 1);
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
    m.begin_with_init();
    res = curl_easy_perform(curl);
    m.end();
  }
  if (formpost != NULL)
    curl_formfree(formpost);

  clean_up();
  *size = get_file_size(full_file_name);
  *time = m.get_papi_elapsed_time();
#endif
}

void CurlUtil::recv_file(char *file_name, char *download_url,
                         int *size, double *time) {
#ifndef INTERFACE_MPI
  curl = curl_easy_init();
  FILE *fp = fopen(file_name, "wb");
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_FILE, fp);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_URL, download_url);
    //curl_easy_setopt(curl, CURLOPT_HEADER, 1);
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
    m.begin_with_init();
    res = curl_easy_perform(curl);
    m.end();
  }
  fclose(fp);

  clean_up();
  *size = get_file_size(file_name);
  *time = m.get_papi_elapsed_time();
#endif
}

void CurlUtil::get(char *url) {
#ifndef INTERFACE_MPI
  curl = curl_easy_init();
  // do a curl GET (blocking)
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    //curl_easy_setopt(curl, CURLOPT_HEADER, 1);
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
    res = curl_easy_perform(curl);
  }

  clean_up();
#endif
}

void CurlUtil::get_nonblocking(char *url) {
#ifndef INTERFACE_MPI
  char cmd[1024];
  sprintf(cmd, "curl --silent \"%s\" &", url);
  int ret = system(cmd);
  //multi_curl = curl_multi_init();
  //curl = curl_easy_init();
  //if (curl && multi_curl)
  //{
  //  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  //  curl_easy_setopt(curl, CURLOPT_URL, url);
  //  //curl_easy_setopt(curl, CURLOPT_HEADER, 1);
  //  //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
  //  //res = curl_easy_perform(curl);
  //  curl_multi_add_handle(multi_curl, curl);
  //  int handles = 1;
  //  curl_multi_perform(multi_curl, &handles);
  //}

  //clean_up();
  //if (multi_curl)
  //  curl_multi_cleanup(multi_curl);
#endif
}

void CurlUtil::clean_up() {
#ifndef INTERFACE_MPI
  if (curl != NULL)
    curl_easy_cleanup(curl);
#endif
}
