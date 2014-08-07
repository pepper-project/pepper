#ifndef CODE_PEPPER_CMTGKR_CIRCUIT_CIRCUIT_DATA_H_
#define CODE_PEPPER_CMTGKR_CIRCUIT_CIRCUIT_DATA_H_

#include <vector>
#include <deque>
#include <string>
#include <sys/stat.h>

#include <common/mpnvector.h>

#define LAYER_DATA_HEADER 0x11111111

template<typename T>
class LayerData
{
private:
  int layer;
  MPNVector<T> gates;

  bool isDirty;

public:
  LayerData(int layerIdx)
    : layer(layerIdx), gates(0), isDirty(false) { }

  LayerData(int layerIdx, size_t layerSize)
    : layer(layerIdx), gates(layerSize), isDirty(false) { }

  T&       operator[](int gateIdx)       { isDirty = true; return gates[gateIdx]; }
  const T& operator[](int gateIdx) const { return gates[gateIdx]; }

  int index() const { return layer; }
  void dirty() { isDirty = true; }

  bool save(const std::string& dir) const;
  bool load(const std::string& dir);

  /*
   * Attempt to load the layer, which should be of the specified size.
   * If no layer was loaded, then create a zero-filled layer of that size.
   */
  void tryLoad(const std::string& dir, size_t size);

  size_t dataSize() const;

protected:
  std::string layerName() const;
  bool shouldSave() const { return isDirty; }
};

template<typename T>
class CircuitData
{
private:
  std::deque<LayerData<T> > blocks;

  std::vector<size_t> blockSizes;
  std::string circuitDir;
  size_t budget;
  size_t usage;

public:
  CircuitData();
  CircuitData(size_t memBudget, const std::vector<size_t>& layerSizes, const std::string& suffix = "");

  LayerData<T>& operator[](int layerIdx);

  void setBudget(size_t newBudget);
  void setSizes(const std::vector<size_t>& newSizes);

  bool save() const;
  LayerData<T>& load(int layerIdx);

protected:
  void updateUsage();
  bool shouldEvict(bool evictToAdd) const;
};

typedef CircuitData<mpz_t> MPZData;
typedef CircuitData<mpq_t> MPQData;

typedef LayerData<mpz_t> LayerMPZData;
typedef LayerData<mpq_t> LayerMPQData;

#endif /* CODE_PEPPER_CMTGKR_CIRCUIT_CIRCUIT_DATA_H_ */
