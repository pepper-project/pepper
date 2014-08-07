#include <apps_sfdl_hw/binary_tree_test_p_exo.h>
#include <apps_sfdl_gen/binary_tree_test_cons.h>
#include <map>
#include <stdint.h>
#include <include/binary_tree.h>

#define MAX_TREE_RESULTS 4

//typedef uint32_t tree_key_t;
//typedef uint32_t tree_value_t;

//typedef struct tree_result {
	//tree_key_t key;
	//tree_value_t value;
//} tree_result_t;

//typedef struct tree_result_set {
	//uint8_t num_results;
	//tree_result_t results[MAX_TREE_RESULTS];
//} tree_result_set_t;

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.


typedef std::multimap<tree_key_t, tree_key_t> Tree;
typedef Tree::iterator TreeItr;
typedef std::pair<TreeItr, TreeItr> TreeRange;
typedef Tree::const_iterator CTreeItr;
typedef std::pair<CTreeItr, CTreeItr> CTreeRange;

static void output_results(mpq_t** outputs, const tree_result_set_t& result_set) {
	mpq_t* out = *outputs;

	mpq_set_ui(*out++, result_set.num_results, 1);
	for (int i = 0; i < MAX_TREE_RESULTS; i++) {
		mpq_set_ui(*out++, result_set.results[i].key, 1);
		mpq_set_ui(*out++, result_set.results[i].value, 1);
	}

	*outputs = out;
}

static void find_results(mpq_t** outputs, CTreeItr start, CTreeItr end) {
	tree_result_set_t result_set;
	memset(&result_set, 0, sizeof(tree_result_set_t));

	for (CTreeItr itr = start; itr != end; ++itr) {
		int i = result_set.num_results;
		result_set.results[i].key = itr->first;
		result_set.results[i].value = itr->second;
		result_set.num_results++;
		if (result_set.num_results >= MAX_TREE_RESULTS) break;
	}

	output_results(outputs, result_set);
}

static void find_eq(mpq_t** outputs, const Tree& t, tree_key_t key) {
	CTreeRange range = t.equal_range(key);
	find_results(outputs, range.first, range.second);
}

static void find_lt(mpq_t** outputs, const Tree& t, tree_key_t key, bool equal_to) {
	CTreeRange range = t.equal_range(key);
	CTreeItr end = (equal_to) ? range.second : range.first;
	find_results(outputs, t.begin(), end);
}

static void find_gt(mpq_t** outputs, const Tree& t, tree_key_t key, bool equal_to) {
	CTreeRange range = t.equal_range(key);
	CTreeItr start = (equal_to) ? range.first : range.second;
	find_results(outputs, start, t.end());
}

static void find_range(mpq_t** outputs, const Tree& t, tree_key_t low_key, bool low_equal_to,
					   tree_key_t high_key, bool high_equal_to) {
	CTreeRange low_range = t.equal_range(low_key);
	CTreeRange high_range = t.equal_range(high_key);
	CTreeItr start = (low_equal_to) ? low_range.first : low_range.second;
	CTreeItr end = (high_equal_to) ? high_range.second : high_range.first;
	find_results(outputs, start, end);
}

static void remove_value(Tree& t, tree_key_t key, tree_value_t value) {
	TreeRange range = t.equal_range(key);
	for (TreeItr itr = range.first; itr != range.second; ++itr) {
		if (itr->second == value) {
			t.erase(itr);
		}
	}
}

static void print_outputs(const mpq_t* output_q, int num_outputs) {
	gmp_printf("\n");
	gmp_printf("binary_tree_test outputs:\n");
	for (int i = 0; i < num_outputs; i++) {
	  gmp_printf("Output %d: %Zd\n", i, mpq_numref(output_q[i]));
	}
	gmp_printf("\n\n");
}


binary_tree_testProverExo::binary_tree_testProverExo() { }

void binary_tree_testProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

	output_recomputed++;
	Tree t;

	t.insert(make_pair(6, 47));

	t.insert(make_pair(5, 42));
	t.insert(make_pair(5, 43));
	t.insert(make_pair(5, 44));

	remove_value(t, 5, 43);

	find_eq(&output_recomputed, t, 5);
	find_eq(&output_recomputed, t, 6);

	t.erase(5);

	find_eq(&output_recomputed, t, 5);

	t.insert(make_pair(9, 112));
	t.insert(make_pair(27, 8));
	t.insert(make_pair(63, 789));

	find_lt(&output_recomputed, t, 27, false);
	find_gt(&output_recomputed, t, 6, false);
	find_range(&output_recomputed, t, 9, true, 63, false);
}

//Refer to apps_sfdl_gen/binary_tree_test_cons.h for constants to use in this exogenous
//check.
bool binary_tree_testProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING

  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

//  gmp_printf("Real output:\n");
//  print_outputs(output_q, num_outputs);
//  gmp_printf("\n");
//
//  gmp_printf("Recomputed output:\n");
//  print_outputs(output_recomputed, num_outputs);
//  gmp_printf("\n");

  for(int i = 0; i < num_outputs; i++){
	  if (!mpq_equal(output_recomputed[i], output_q[i])){
		  passed_test = false;
		  break;
      }
  }

  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

