#include <common/utility.h>
#include <mysql.h>
#include <pthread.h>
#include <stdint.h>
#include <storage/db_util.h>

#include <apps_sfdl/student_db.h>
#include <include/db.h>

enum stage {
  CREATE_DB = 0,
  READONLY_QUERY = 1,
  READWRITE_QUERY = 2,
};

void finish_with_error(MYSQL *conn)
{
  fprintf(stderr, "Error: %s\n", mysql_error(conn));
  mysql_close(conn);
  exit(1);
}

MYSQL* open_db(const char* db_name) {
  printf("Opening database.\n");
  MYSQL *conn = mysql_init(NULL);
  if (conn == NULL) {
    fprintf(stderr, "%s\n", mysql_error(conn));
    exit(1);
  }

  if (mysql_real_connect(conn, "localhost", "root", "", NULL, 0, NULL, 0) == NULL) {
    fprintf(stderr, "%s\n", mysql_error(conn));
    exit(1);
  }

  if (db_name != NULL) {
    if (mysql_select_db(conn, db_name)) {
      finish_with_error(conn);
    }
  }
  printf("Database opened.\n");
  return conn;
}

void create_db(const char* database, int number_of_rows, bool readonly) {
  MYSQL* conn = open_db(NULL);
  char query_template[BUFLEN];


  snprintf(query_template, BUFLEN - 1, "DROP DATABASE IF EXISTS %s", database);
  printf("Dropping existed database.\n");
  if (mysql_query(conn, query_template)) {
    finish_with_error(conn);
  }
  printf("Existed database dropped.\n");

  snprintf(query_template, BUFLEN - 1, "CREATE DATABASE %s", database);
  printf("Creating empty DB.\n");
  if (mysql_query(conn, query_template)) {
    finish_with_error(conn);
  }

  if (mysql_select_db(conn, database)) {
    finish_with_error(conn);
  }

  if (mysql_query(conn, "CREATE TABLE Student (ID INT PRIMARY KEY, FName BIGINT, LName BIGINT, Age INT, Major INT, State INT, PhoneNum INT, Class INT, Credits INT, Average INT, Honored INT)")) {
    finish_with_error(conn);
  }

  printf("Table created.\n");

  int* average_keys = new int[number_of_rows];
  int* class_keys = new int[number_of_rows];
  int* age_keys = new int[number_of_rows];

  srand(time(NULL));
  generate_random_permutation(average_keys, number_of_rows);
  generate_random_permutation(class_keys, number_of_rows);
  generate_random_permutation(age_keys, number_of_rows);

  for (int i = 0; i < number_of_rows; i++) {
    if (i % 100 == 0) {
      printf("%d rows inserted.\n", i);
    }
    Student_t tempStudent;

    tempStudent.KEY = i;
    tempStudent.FName = 1298384231432L + rand();
    tempStudent.LName = 2380943023039L + rand();
    tempStudent.Age = age_keys[i];
    tempStudent.Major = 10 + (rand() % 30);
    tempStudent.State = rand() % 50;
    tempStudent.PhoneNum = 512800000 + rand();
    tempStudent.Class = class_keys[i];
    tempStudent.Credits = 100 + rand();
    tempStudent.Average = average_keys[i];
    tempStudent.Honored = 0;

    snprintf(query_template, BUFLEN - 1, "INSERT INTO Student (ID, FName, \
      LName, Age, Major, State, PhoneNum, Class, Credits, Average, Honored) \
        VALUES (%d, %lld, %lld, %d, %d, %d, %d, %d, %d, %d, %d)", tempStudent.KEY,
        tempStudent.FName, tempStudent.LName, tempStudent.Age,
        tempStudent.Major, tempStudent.State, tempStudent.PhoneNum,
        tempStudent.Class, tempStudent.Credits, tempStudent.Average,
        tempStudent.Honored);

    if (mysql_query(conn, query_template)) {
      finish_with_error(conn);
    }
  }

  if (mysql_query(conn, "CREATE INDEX Average_idx ON Student (Average)")) {
    finish_with_error(conn);
  }
  if (readonly) {
    if (mysql_query(conn, "CREATE INDEX Class_idx ON Student (Class)")) {
      finish_with_error(conn);
    }
    if (mysql_query(conn, "CREATE INDEX Age_idx ON Student (Age)")) {
      finish_with_error(conn);
    }
  }

  mysql_close(conn);
  delete[] average_keys;
  delete[] class_keys;
  delete[] age_keys;
}

void measure_readonly_query(const char* database, const char* query, int samples) {
  MYSQL* conn = open_db(database);

  for (int i = 0; i < samples; i++) {
    if (i % 10000 == 0) {
      printf("%d queries executed.\n", i);
    }
    if (mysql_query(conn, query)) {
      finish_with_error(conn);
    }

    int status;
    MYSQL_RES *result = mysql_store_result(conn);
    if (result == NULL) {
      finish_with_error(conn);
    }
    do {
      MYSQL_ROW row = mysql_fetch_row(result);
      if (row == NULL) {
        break;
      }
    } while(true);

    mysql_free_result(result);
  }
  mysql_close(conn);
}

void measure_readwrite_query(const char* database, const char* query, int samples) {
  MYSQL* conn = open_db(database);

  if (mysql_query(conn, "DROP TABLE IF EXISTS HonoredStudent")) {
    finish_with_error(conn);
  }
  if (mysql_query(conn, "CREATE TABLE HonoredStudent (ID INT, FName BIGINT, LName BIGINT, Major INT, Average INT)")) {
    finish_with_error(conn);
  }

  for (int i = 0; i < samples; i++) {
    if (i % 10000 == 0) {
      printf("%d queries executed.\n", i);
    }
    if (mysql_query(conn, query)) {
      finish_with_error(conn);
    }
    MYSQL_RES *result = mysql_store_result(conn);
    mysql_free_result(result);
  }

  mysql_close(conn);
}

int main(int argc, char **argv) {
  //printf("MySQL client version: %s\n", mysql_get_client_info());

  //char query[] = "SELECT * FROM Student WHERE Average > 90 LIMIT 5";
  //char query[] = "SELECT * FROM Student WHERE Class = 2009 LIMIT 5";
  //char query[] = "SELECT * FROM Student WHERE Age > 20 AND Age < 24 LIMIT 5";
  //char query[] = "UPDATE Student SET Honored = 1 WHERE Average > 90 LIMIT 1";

  int op = atoi(argv[1]);
  switch (op) {
    case CREATE_DB:
      {
        const char* database = argv[2];
        int number_of_rows = atoi(argv[3]);
        bool readonly = false;
        if (!strcmp(argv[4], "readonly")) {
          readonly = true;
        }
        create_db(database, number_of_rows, readonly);
        break;
      }
    case READONLY_QUERY:
      {
        const char* database = argv[2];
        const char* query = argv[3];
        int samples = atoi(argv[4]);
        measure_readonly_query(database, query, samples);
        break;
      }
    case READWRITE_QUERY:
      {
        const char* database = argv[2];
        const char* query = argv[3];
        int samples = atoi(argv[4]);
        measure_readwrite_query(database, query, samples);
        break;
      }
    default:
      break;
  }

  return 0;
}
