#include <sstream>

#include <common/measurement.h>
#include <libv/input_creator.h>

InputCreator::InputCreator() {
  char bstore_folder_path[BUFLEN];
  snprintf(bstore_folder_path, BUFLEN - 1, "%s/block_stores", FOLDER_STATE);
  mkdir(bstore_folder_path, S_IRWXU);
}

void InputCreator::set_block_store_name(const char *shared_block_store_name) {
  shared_bstore_file_name = string(shared_block_store_name);
  if (shared_bstore_file_name == "") {
    std::ostringstream oss;
    oss << "unnamed_block_store_" << PAPI_get_real_nsec();
    shared_bstore_file_name = oss.str();
  }
}


