#include <cassert>
#include <sstream>
#include <stdint.h>
#include <iostream>
#include <cerrno>
#include <stdexcept>

#include <crypto/prng.h>

#include <common/utility.h>
#include <common/mpnops.h>
#include <common/debug_utils.h>

#include "circuit_data.h"

using namespace std;

static Prng prng(PNG_CHACHA);

#define RETURN_IF_FALSE(file, statement) { if (!statement) { fclose(f); return false; } }

template<typename T> bool
write(FILE* f, const T& val)
{
  return fwrite(&val, sizeof(val), 1, f) == 1;
}

template<> bool
write<mpz_t>(FILE* f, const mpz_t& val)
{
  return mpz_out_raw(f, val) != 0;
}

template<> bool
write<mpq_t>(FILE* f, const mpq_t& val)
{
  return (mpz_out_raw(f, mpq_numref(val)) != 0) && (mpz_out_raw(f, mpq_denref(val)) != 0);
}

template<typename T> bool
read(T& val, FILE* f)
{
  return fread(&val, sizeof(val), 1, f) == 1;
}

template<> bool
read<mpz_t>(mpz_t& val, FILE* f)
{
  return mpz_inp_raw(val, f) != 0;
}

template<> bool
read<mpq_t>(mpq_t& val, FILE* f)
{
  return (mpz_inp_raw(mpq_numref(val), f) != 0) && (mpz_inp_raw(mpq_denref(val), f) != 0);
}

template<typename T> size_t
sizeOf(const T& obj)
{
  return sizeof(obj);
}

template<> size_t
sizeOf<mpz_t>(const mpz_t& val)
{
  return mpz_sizeinbase(val, 2) / 8 + sizeof(val);
}

template<> size_t
sizeOf<mpq_t>(const mpq_t& val)
{
  return sizeOf(mpq_numref(val)) + sizeOf(mpq_denref(val));
}

template<typename T> T
getRandom(Prng& prng)
{
  T out;
  prng.get_randomb(reinterpret_cast<char*>(&out), sizeof(out) * 8);
  return out;
}

static FILE *
open_file(const string& filename, const string& mode, const string& directory, bool createDir = true)
{
  if (createDir)
    recursive_mkdir(directory);

  string fname(directory);
  fname += '/';
  fname += filename;
  return fopen(fname.c_str(), mode.c_str());
}

template<typename T>
bool LayerData<T>::
save(const string& directory) const
{
  if (shouldSave())
  {
    string name = layerName();
    FILE* f;
    if (!(f = open_file(layerName(), "wb", directory)))
      return false;

    RETURN_IF_FALSE(f, write(f, LAYER_DATA_HEADER));
    RETURN_IF_FALSE(f, write(f, gates.size()));
    for (size_t i = 0; i < gates.size(); i++)
      RETURN_IF_FALSE(f, write(f, gates[i]));

    fflush(f);
    fclose(f);
  }
  return true;
}

template<typename T>
bool LayerData<T>::
load(const string& directory)
{
  string name = layerName();
  FILE* f;
  uint32_t header;
  size_t size;

  if (!(f = open_file(layerName(), "rb", directory, false)))
    return false;

  RETURN_IF_FALSE(f, read(header, f));
  RETURN_IF_FALSE(f, read(size, f));
  gates.resize(size);
  isDirty = false;

  for (size_t i = 0; i < gates.size(); i++)
    RETURN_IF_FALSE(f, read(gates[i], f));

  fclose(f);
  return true;
}

template<typename T>
void LayerData<T>::
tryLoad(const string& directory, size_t size)
{
  if (!load(directory))
  {
    if (gates.size() != size)
      gates.resize(size);

    assert(gates.size() == size);
    for (size_t i = 0; i < size; i++)
      mpn_ops<T>::init(gates[i]);
  }
}

template<typename T>
size_t LayerData<T>::
dataSize() const
{
  double avg = 0;

  const uint32_t numSamples = 4;
  if (gates.size() > 0)
  {
    for (uint32_t i = 0; i < numSamples; i++)
    {
      size_t idx = getRandom<size_t>(prng) % gates.size();
      avg = sizeOf(gates[idx]);
    }

    avg /= numSamples;
  }

  return size_t(avg * gates.size()) + sizeof(this);
}

template<typename T>
string LayerData<T>::
layerName() const
{
  stringstream ss;
  ss << "layer_" << index();
  return ss.str();
}

template<typename T> string
circuitPrefix()
{ //TODO: Use static_assert.
  throw runtime_error("Does not work.");
}

template<> string
circuitPrefix<mpz_t>()
{ return "circuit"; }

template<> string
circuitPrefix<mpq_t>()
{ return "circuitq"; }

template<typename T>
CircuitData<T>::
CircuitData(size_t memBudget, const std::vector<size_t>& sizes, const string& suffix)
  : blocks(), blockSizes(sizes), circuitDir(), budget(memBudget), usage(0)
{
  stringstream ss;

  ss << FOLDER_STATE << "/" << circuitPrefix<T>() << "_";
  if (suffix.empty())
  {
    struct stat sb;
    do
    {
      stringstream testDir;
      uint64_t randNumber = getRandom<uint64_t>(prng);

      testDir << ss.str() << hex << randNumber;
      circuitDir = testDir.str();

    } while (stat(circuitDir.c_str(), &sb) == 0);
  }
  else
  {
    ss << suffix;
    circuitDir = ss.str();
  }

  // Reserve circuit directory.
  if (!recursive_mkdir(circuitDir))
    throw runtime_error("Could not create new directory for circuit.");
}

template<typename T>
LayerData<T>& CircuitData<T>::
operator [](int layerIdx)
{
  assert(inRange<int>(layerIdx, 0, blockSizes.size()));
  typedef typename deque<LayerData<T> >::iterator LayerDataIt;
  for (LayerDataIt it = blocks.begin(); it != blocks.end(); ++it)
  {
    if (it->index() == layerIdx)
      return *it;
  }

  LayerData<T>& newLayer = load(layerIdx);
  assert(newLayer.index() == layerIdx);
  return newLayer;
}

template<typename T>
void CircuitData<T>::
setBudget(size_t newBudget)
{
  budget = newBudget;
}

template<typename T>
void CircuitData<T>::
setSizes(const std::vector<size_t>& newSizes)
{
  this->blockSizes = newSizes;
}

template<typename T>
bool CircuitData<T>::
save() const
{
  bool success = true;
  typedef typename deque<LayerData<T> >::const_iterator LayerDataCIt;
  for (LayerDataCIt it = blocks.begin(); it != blocks.end(); ++it)
    success = it->save(circuitDir) && success;
  return success;
}

template<typename T>
LayerData<T>& CircuitData<T>::
load(int layerIdx)
{
  assert(inRange<int>(layerIdx, 0, blockSizes.size()));

  updateUsage();
  while (shouldEvict(true))
  {
    LayerData<T>& layer = blocks.front();
    assert(usage >= layer.dataSize());

    //cout << "Evicting Layer: " << layer.index() << " Usage: " << usage << endl;

    layer.save(circuitDir);
    usage -= layer.dataSize();
    blocks.pop_front();
  }

  //cout << "Load Layer: " << layerIdx << " Usage: " << usage << endl;
  blocks.push_back(LayerData<T>(layerIdx));
  LayerData<T>& newLayer = blocks.back();
  newLayer.tryLoad(circuitDir, blockSizes[layerIdx]);
  usage += newLayer.dataSize();

  return newLayer;
}

template<typename T>
void CircuitData<T>::
updateUsage()
{
  usage = 0;
  typedef typename deque<LayerData<T> >::const_iterator LayerDataCIt;
  for (LayerDataCIt it = blocks.begin(); it != blocks.end(); ++it)
    usage += it->dataSize();
}

template<typename T>
bool CircuitData<T>::
shouldEvict(bool evictToAdd) const
{
  size_t minNumLayers = 2 + (evictToAdd ? -1 : 0);
  return (minNumLayers < blocks.size()) && (usage > budget);
}

template class CircuitData<mpz_t>;
template class CircuitData<mpq_t>;
