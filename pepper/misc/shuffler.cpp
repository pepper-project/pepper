#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>

#define BUFLEN 10240

using namespace std;

void parse_args(int argc, char **argv, int *num_mappers,
  int *num_reducers, int *size_input_red, char *folder_path) {
  
  if (argc < 4) {
    cout<<"LOG: Fewer arguments passed"<<endl;
    exit(1);
  }

  if (folder_path != NULL) 
    folder_path[0] = '\0';

  for (int i=1; i<argc; i++) {
    if (strcmp(argv[i], "-m") == 0 && num_mappers != NULL)
      *num_mappers = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-r") == 0 && num_reducers != NULL)
      *num_reducers = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-i") == 0 && size_input_red != NULL)
      *size_input_red = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-f") == 0 && folder_path != NULL) {
      strncpy(folder_path, argv[i+1], BUFLEN-1);
      folder_path[BUFLEN-1] = '\0';
    }
  }
}

int main(int argc, char **argv) {
  int num_mappers, num_reducers, size_input_red;
  char folder_path[BUFLEN];
  char scratch_str[BUFLEN];

  parse_args(argc, argv, &num_mappers, &num_reducers, &size_input_red, folder_path);
  
  FILE *fp_m[num_mappers];
  FILE *fp_m_q[num_mappers];
  FILE *fp_r[num_reducers];
  FILE *fp_r_q[num_reducers];

  for (int i=0; i<num_mappers; i++) {
    snprintf(scratch_str, BUFLEN-1, "%s/output_b_%d", folder_path, i);
    fp_m[i] = fopen(scratch_str, "r");
    snprintf(scratch_str, BUFLEN-1, "%s/output_q_b_%d", folder_path, i);
    fp_m_q[i] = fopen(scratch_str, "r");
  }

  for (int i=0; i<num_reducers; i++) {
    snprintf(scratch_str, BUFLEN-1, "%s/input_b_%d", folder_path, i);
    fp_r[i] = fopen(scratch_str, "w");
    snprintf(scratch_str, BUFLEN-1, "%s/input_q_b_%d", folder_path, i);
    fp_r_q[i] = fopen(scratch_str, "w");
  }

  int size_input_each_mapper =  size_input_red/num_mappers;
  mpz_t temp;
  mpq_t temp_q;

  mpz_init(temp);
  mpq_init(temp_q);
  
  for (int k=0; k<num_reducers; k++) {
    for (int i=0; i<num_mappers; i++) {
      for (int j=0; j<size_input_each_mapper; j++) {
        mpz_inp_raw(temp, fp_m[i]);
        mpz_out_raw(fp_r[k], temp); 

        mpz_inp_raw(mpq_numref(temp_q), fp_m_q[i]);
        mpz_inp_raw(mpq_denref(temp_q), fp_m_q[i]);
        
        mpz_out_raw(fp_r_q[k], mpq_numref(temp_q)); 
        mpz_out_raw(fp_r_q[k], mpq_numref(temp_q)); 
      }
    }
  }

  for (int i=0; i<num_mappers; i++) {
    fclose(fp_m[i]);
    fclose(fp_m_q[i]);
  }

  for (int i=0; i<num_reducers; i++) {
    fclose(fp_r[i]);
    fclose(fp_r_q[i]);
  }  

  mpz_clear(temp);
  mpq_clear(temp_q);
  return 0;
}
