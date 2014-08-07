#include <fstream>
#include <iostream>
#include <limits>

#include <common/math.h>

#include "magic_var_operation.h"
#include "pws_circuit_parser.h"

using namespace std;

static const bool DEBUG_MODE = true;
bool a = false;

typedef map<string, int>::iterator ConstMapIt;
typedef map<int, CVar*>::iterator CVarIt;
typedef vector<CConst*>::iterator CConstIt;

#define ASSERT_RETURN(pws, expected, parseContext)    \
  if (!assert(pws, expected, parseContext))           \
    return;

#define IF_INVALID_RETURN(maybe)  if (!maybe.isValid()) return;

static void
parseError(const string& msg)
{
  if (DEBUG_MODE)
    cerr << "PARSE ERROR - " << msg << endl;
}

static bool
assert(Tokenizer& pws, string expected, string parseCtx)
{
  string token;
  pws >> token;

  if (token != expected){
    stringstream ss("Missing '");
    ss << expected << "' in " << parseCtx << ".";
    parseError(ss.str());
    return false;
  }
  return true;
}

template<class T> void
clearVector(vector<T>& vec)
{
  typename vector<T>::iterator it;
  for (it = vec.begin(); it != vec.end(); ++it)
    delete *it;
  vec.clear();
}

template<class T1, class T2> void
clearPairVector(vector< pair<T1, T2*> >& vec)
{
  typename vector< pair<T1, T2*> >::iterator it;
  for (it = vec.begin(); it != vec.end(); ++it)
    delete it->second;
  vec.clear();
}

template<class T1, class T2> void
clearMap(map<T1, T2*>& m)
{
  typename map<T1, T2*>::iterator it;
  for (it = m.begin(); it != m.end(); ++it)
    delete it->second;
  m.clear();
}

PWSCircuitParser::
PWSCircuitParser(const mpz_t p)
  : varMap(), inOutVarsMap(), outConstantDesc(), constants(),
    constMap(), token(), circuitDesc(), outputGateBegin(0)
{
  mpz_init(prime);
  mpz_set(prime, p);
}

void PWSCircuitParser::
clear()
{
  clearPrivate();
  clearIndexes();

  circuitDesc.clear();
  clearPairVector(magicOps);

  opCount.clear();
}

const CircuitDescription& PWSCircuitParser::
getCircuitDescription() const
{
  return circuitDesc;
}

void PWSCircuitParser::
parse(const string& pwsFileName)
{
  clear();

  ifstream pwsFile(pwsFileName.c_str());
  if (!pwsFile.is_open()){
    parseError("Couldn't open pws file: " + pwsFileName);
    return;
  }

  int i =0;
  Tokenizer pws(pwsFile);
  while (pws.hasNext())
  {
    pws >> token;

    /*
    if (i ++ % 512 == 0)
    {
      cout << endl << "========" << endl;
      cout << i << endl;
      printMemoryStats();
    }
    */

    if (token == "P")
      parsePolyConstraint(pws);
    else if (token == "<I")
      parseLessThanInt(pws);
    else if (token == "<F")
      parseLessThanFloat(pws);
    else if (token == "!=")
      parseNotEqual(pws);
    else if (token == "/")
      parseDivide(pws);
    else if (token[0] == '/' && token[1] == '/')
      pws.ignoreLine();
  }
  pwsFile.close();

  makeOutputLayer();
  constructInvertedIndexes();

  //printMemoryStats();

  clearPrivate();
}

void PWSCircuitParser::
makeOutputLayer()
{
  outputGateBegin = numeric_limits<int>::max();
  int outputLayer = 0;
  typedef vector<ConstOutputDescription>::iterator COutIt;
  for (COutIt it = outConstantDesc.begin();
       it != outConstantDesc.end();
       ++it)
  {
    outputLayer = max(it->first.layer, outputLayer);
  }

  int numOutputs = 0;
  for (CVarIt it = inOutVarsMap.begin(); it != inOutVarsMap.end(); ++it)
  {
    const CVar* cVar = it->second;
    if (cVar->isOutput() && cVar->isBound())
    {
      outputGateBegin = min(outputGateBegin, cVar->name);
      outputLayer = max(cVar->minLayer, outputLayer);
    }
  }

  //cout << outputLayer << endl;
  //cout << "NUMOutCONS: " << outConstantDesc.size() << endl;
  for (COutIt it = outConstantDesc.begin();
       it != outConstantDesc.end();
       ++it)
  {
    GatePosition newPos = promoteGate(it->first, outputLayer);
    outConstants[it->second].push_back(newPos.name);
  }

  //cout << "Promoted outputs." << endl;
  for (CVarIt it = inOutVarsMap.begin(); it != inOutVarsMap.end(); ++it)
  {
    CVar* cVar = it->second;
    if (cVar->isOutput() && cVar->isBound())
    {
      promotePrimitive(*cVar, outputLayer);
    }
  }
}

void PWSCircuitParser::
constructInvertedIndexes()
{
  // IO Vars
  for (CVarIt it = inOutVarsMap.begin(); it != inOutVarsMap.end(); ++it)
  {
    CVar* cVar = it->second;
    if (cVar->isBound())
    {
      if (cVar->isOutput())
        outGates[cVar->name] = cVar->gateIndex.back();
      else if (cVar->isInput())
        inGates[cVar->name] = cVar->gateIndex.front();
    }
  }

  // Magic Vars
  for (CVarIt it = varMap.begin(); it != varMap.end(); ++it)
  {
    CVar* cVar = it->second;
    if (cVar->isBound() && cVar->isMagic())
      magicGates.push_back(cVar->gateIndex.front());
  }

  // Input Constants
  for (CConstIt it = constants.begin(); it != constants.end(); ++it)
  {
    CConst* cons = *it;
    if (cons->isBound())
      inConstants.push_back(pair<string, int>(cons->value, cons->gateIndex.front()));
  }
}

LayerDescription& PWSCircuitParser::
getLayer(int layerNum)
{
  while (circuitDesc.size() <= (size_t) layerNum)
    circuitDesc.push_back(LayerDescription());

  return circuitDesc[layerNum];
}

GateDescription& PWSCircuitParser::
getGate(const GatePosition pos)
{
  return getLayer(pos.layer)[pos.name];
}

GatePosition PWSCircuitParser::
getGate(CPrimitive& p, int layer)
{
  addToCircuit(p);

  if (layer == 0)
    layer = p.minLayer;

  return promotePrimitive(p, layer);
}

GatePosition PWSCircuitParser::
getGate(const Primitive& p, int layer)
{
  return getGate(getCircuitPrimitive(p), layer);
}

CPrimitive& PWSCircuitParser::
getCircuitPrimitive(const Primitive& p)
{
  if (p.isConstant())
    return *getCircuitConstant(p).value();
  else
    return *getCircuitVariable(p).value();
}

Maybe<CVar*> PWSCircuitParser::
getCircuitVariable(const Primitive& p)
{
  Maybe<CVar*> out;
  if (p.isVariable())
    out = getVarMap(p)[p.idx];
  return out;
}

Maybe<CConst*> PWSCircuitParser::
getCircuitConstant(const Primitive& p)
{
  Maybe<CConst*> out;
  if (p.isConstant())
    out = constants[p.idx];
  return out;
}

void PWSCircuitParser::
getOpGuard(vector<int>& guard)
{
  guard.clear();
  guard.reserve(circuitDesc.size());

  CircuitDescription::const_iterator it;
  for (it = circuitDesc.begin(); it != circuitDesc.end(); ++it)
  {
    guard.push_back(it->size());
  }
}

map<int, CVar*>& PWSCircuitParser::
getVarMap(const Primitive& p)
{
  if (p.isInputOutput())
    return inOutVarsMap;
  else
    return varMap;
}

void PWSCircuitParser::
clearPrivate()
{
  clearMap(varMap);
  clearMap(inOutVarsMap);

  outConstantDesc.clear();

  clearVector(constants);
  constMap.clear();

  // Unbound all gates now that we've cleared the primitives they were bound to.
  typedef CircuitDescription::iterator LayerIt;
  typedef LayerDescription::iterator GateIt;
  for (LayerIt lit = circuitDesc.begin(); lit != circuitDesc.end(); ++lit)
  {
    LayerDescription& layer = *lit;
    for (GateIt git = layer.begin(); git != layer.end(); ++git)
      git->unbind();
  }
}

void PWSCircuitParser::
clearIndexes()
{
  outConstants.clear();
  inConstants.clear();
  inGates.clear();
  outGates.clear();
  magicGates.clear();
}

GatePosition PWSCircuitParser::
addGate(GateDescription::OpType op, int in1, int in2, int outLayerNum)
{
  LayerDescription& outLayer = getLayer(outLayerNum);

  GateDescription gOut(op, outLayerNum, outLayer.size(), in1, in2);

  outLayer.push_back(gOut);

  return gOut.pos;
}

void PWSCircuitParser::
addToCircuit(CPrimitive& p)
{
  if (!p.isBound())
    bindVariable(p.toPrimitive(), addGate(GateDescription::CONSTANT, 0, 0, 0));
}

CVar& PWSCircuitParser::
addVariable(CVar::VarType type, int name)
{
  CVar testVar(type, name);

  map<int, CVar*>& vMap = getVarMap(testVar.toPrimitive());

  map<int, CVar*>::iterator it = vMap.find(name);
  if (it == vMap.end())
  {
    vMap[name] = new CVar(testVar);;
  }

  return *vMap[name];
}

CConst& PWSCircuitParser::
addConstant(mpz_t constant)
{
  char* val;
  int numChars = gmp_asprintf(&val, "%Zd", constant);
  string constStr(val);

  CConst& con = addConstant(constStr);

  void (*freefunc) (void *, size_t);
  mp_get_memory_functions (NULL, NULL, &freefunc);
  freefunc(val, (numChars + 1) * sizeof(char));

  return con;
}

CConst& PWSCircuitParser::
addConstant(int constant)
{
  stringstream ss;
  ss << constant;

  return addConstant(ss.str());
}

CConst& PWSCircuitParser::
addConstant(const string& constant)
{
  ConstMapIt it = constMap.find(constant);
  if (it == constMap.end())
  {
    constants.push_back(new CConst(constant, constants.size()));
    constMap[constant] = constants.size() - 1;
  }
  return *constants[constMap[constant]];
}

void PWSCircuitParser::
addMagicOp(MagicVarOperation* op, vector<int>& guard)
{
  magicOps.push_back(pair<vector<int>, MagicVarOperation*>(guard, op));
}

GatePosition PWSCircuitParser::
promotePrimitive(CPrimitive& p, int layer)
{
  while (layer > p.maxLayer)
  {
    CPrimitive& zero = addConstant(0);
    GatePosition g = makeGate(GateDescription::ADD, p, zero, p.maxLayer);

    getGate(g).bind(p.toPrimitive());
    p.gateIndex.push_back(g.name);
    p.maxLayer++;
  }
  return GatePosition(layer, p.gateIndex[layer - p.minLayer]);
}

GatePosition PWSCircuitParser::
promoteGate(const GatePosition g, int layer)
{
  Maybe<Primitive> boundPrimitive = getGate(g).boundPrimitive;

  if (boundPrimitive.isValid())
  {
    CPrimitive& p = getCircuitPrimitive(boundPrimitive.value());
    return promotePrimitive(p, layer);
  }

  GatePosition gatePos = g;
  while (layer > gatePos.layer)
  {
    CPrimitive& zero = addConstant(0);
    GatePosition zeroGate = getGate(zero, gatePos.layer);
    gatePos = addGate(GateDescription::ADD, gatePos.name, zeroGate.name, gatePos.layer + 1);
  }
  return gatePos;
}

void PWSCircuitParser::
bindVariable(const Primitive& var, const GatePosition gate)
{
  CPrimitive& v = getCircuitPrimitive(var);

  if (v.isBound())
  {
    int layer = v.minLayer;
    vector<int>::iterator it = v.gateIndex.begin();
    for (; it != v.gateIndex.end(); ++it)
    {
      getGate(GatePosition(layer, *it)).unbind();
    }
  }

  getCircuitPrimitive(var).bind(gate);
  getGate(gate).bind(var);
}

void PWSCircuitParser::
bindOutputConstant(const GatePosition outGate, int val)
{
  stringstream ss;
  ss << val;

  ConstOutputDescription desc(outGate, ss.str());
  outConstantDesc.push_back(desc);
}

void PWSCircuitParser::
bindOutputConstant(const Primitive val, const GatePosition outGate)
{
  Maybe<CConst*> cConst = getCircuitConstant(val);
  if (cConst.isValid())
  {
    ConstOutputDescription desc(outGate, cConst.value()->value);
    outConstantDesc.push_back(desc);
  }
}

Primitive PWSCircuitParser::
makeMagic(const Primitive var)
{
  Maybe<CVar*> cVar = getCircuitVariable(var);
  if (!cVar.isValid())
    return var;

  cVar.value()->makeMagic();
  return var;
}

GatePosition PWSCircuitParser::
makeGate(const GateDescription::OpType op, const GatePosition in1, const GatePosition in2)
{
  if (in1.layer != in2.layer)
  {
    int layer = max(in1.layer, in2.layer);

    GatePosition gate1 = promoteGate(in1, layer);
    GatePosition gate2 = promoteGate(in2, layer);

    return addGate(op, gate1.name, gate2.name, layer + 1);
  }
  else
  {
    return addGate(op, in1.name, in2.name, in1.layer + 1);
  }
}

GatePosition PWSCircuitParser::
makeGate(const GateDescription::OpType op, CPrimitive& in1, CPrimitive& in2, int inLayerNum)
{
  GatePosition g1 = getGate(in1, inLayerNum);
  GatePosition g2 = getGate(in2, inLayerNum);

  return addGate(op, g1.name, g2.name, inLayerNum + 1);
}

GatePosition PWSCircuitParser::
makeGate(const GateDescription::OpType op, const Primitive& p1, const Primitive& p2)
{
  CPrimitive& in1 = getCircuitPrimitive(p1);
  CPrimitive& in2 = getCircuitPrimitive(p2);

  return makeGate(op, in1, in2, in1.minMatchingLayer(in2));
}

GatePosition PWSCircuitParser::
makeSubGate(const GatePosition in1, const GatePosition in2)
{
  return makeGate(GateDescription::ADD, in1, negateGate(in2));
}

GatePosition PWSCircuitParser::
makeZeroOrOtherGate(CPrimitive& var, mpz_t other)
{
  mpz_t tmp;
  mpz_init_set(tmp, other);
  mpz_neg(tmp, tmp);

  GatePosition otherConstGate = getGate(addConstant(tmp));
  GatePosition varGate = getGate(var);

  GatePosition nzTerm = makeGate(GateDescription::ADD, otherConstGate, varGate);
  GatePosition zOrOtherGate = makeGate(GateDescription::MUL, varGate, nzTerm);

  bindOutputConstant(zOrOtherGate, 0);

  mpz_clear(tmp);
  return zOrOtherGate;
}

GatePosition PWSCircuitParser::
negatePrimitive(const Primitive& p)
{
  if (p.isConstant())
  {
    string constant = constants[p.idx]->value;
    string negConstant;
    if (constant[0] == '-')
      negConstant = constant.substr(1);
    else
      negConstant = "-" + constant;

    return getGate(addConstant(negConstant));
  }
  else
  {
    return negateGate(getGate(addConstant(-1)));
  }
}

GatePosition PWSCircuitParser::
negateGate(const GatePosition g)
{
  GatePosition neg1Gate = getGate(addConstant(-1), g.layer);

  return addGate(GateDescription::MUL, g.name, neg1Gate.name, g.layer + 1);
}

Maybe<Primitive> PWSCircuitParser::
parsePrimitive(Tokenizer& pws)
{
  pws >> token;
    
  if ((token[0] >= '0' && token[0] <= '9') || token[0] == '-')
    return parseConst(token);
  else
    return parseVar(token);
}

GatePosition PWSCircuitParser::
reduce(const GateDescription::OpType op, const vector<GatePosition>& terms)
{
  if (terms.empty())
    return getGate(addConstant(0));

  if (terms.size() == 1)
    return terms[0];

  vector<GatePosition> in;
  vector<GatePosition> out;

  typedef vector<GatePosition>::const_iterator GDescIt;

  int inLayer = 0;
  for (GDescIt it = terms.begin(); it != terms.end(); ++it)
    inLayer = max(inLayer, it->layer);

  for (GDescIt it = terms.begin(); it != terms.end(); ++it)
    in.push_back(promoteGate(*it, inLayer));

  while (in.size() > 1)
  {
    out.clear();

    size_t startIdx = 0;
    if ((in.size() & 1) == 1)
    {
      out.push_back(promoteGate(in[0], inLayer + 1));
      startIdx = 1;
    }

    for (size_t i = startIdx; i < in.size(); i += 2)
      out.push_back(addGate(op, in[i].name, in[i+1].name, inLayer + 1));

    in.swap(out);
    inLayer++;
  }

  // Careful about returning local references
  return in[0];
}

Maybe<Primitive> PWSCircuitParser::
parseVar(const string& token)
{
  Maybe<Primitive> p;

  if (token.empty())
  {
    parseError("empty variable.");
    p.invalidate();
    return p;
  }

  Maybe<CVar::VarType> varType;
  switch(token[0])
  {
    case 'I':
      varType = CVar::INPUT;
      break;
    case 'O':
      varType = CVar::OUTPUT;
      break;
    case 'V':
      varType = CVar::TEMP;
      break;
    case 'M':
      varType = CVar::MAGIC;
      break;
    default:
      varType.invalidate();
      break;
  }

  if (varType.isValid())
  {
    if (token.size() <= 1)
    {
      parseError("variable: " + token);
      p.invalidate();
    }
    else
    {
      int name = atoi(token.c_str() + 1);
      if (name < 0)
      {
        parseError("variable: " + name);
        p.invalidate();
      }
      else
      {
        CVar& var = addVariable(varType.value(), name);
        p = var.toPrimitive();
      }
    }
  }

  return p;
}

Maybe<Primitive> PWSCircuitParser::
parseConst(const string& token)
{
  // First test that it's a valid constant.
  mpq_t tmp;
  mpq_init(tmp);
  int idx = -1;

  Maybe<Primitive> p;
  if (mpq_set_str(tmp, token.c_str(), 10) != 0)
  {
    parseError(token + " is not a valid constant.");
    p.invalidate();
  }
  else
  {
    int idx = addConstant(token).name;
    p.validate();
    p.value().makeConstant(idx);
  }

  mpq_clear(tmp);

  return p;
}

Maybe<GatePosition> PWSCircuitParser::
parsePoly(Tokenizer& pws, const string end)
{
  bool abort = false;
  vector<GatePosition> terms;

  //cout << "[ParsePoly] begin END: " << end << endl;
  while (pws.hasNext())
  {
    pws >> token;

    //cout << "[ParsePoly] TOKEN: " << token << endl;
    if (token == "+")
    {
      if (terms.empty())
      {
        parseError("Unexpected '+' in polynomial definition.");
        abort = true;
        break;
      }
      //cout << "[ParsePoly] NEW TERM" << endl;
    }
    else if (token == end)
    {
      //cout << "[ParsePoly] TOKEN == END" << endl;
      break;
    }
    else
    {
      pws.rewind();
      Maybe<GatePosition> term = parsePolyTerm(pws);
      if (term.isValid())
        terms.push_back(term.value());
    }
  }

  if (abort || terms.empty())
    return Maybe<GatePosition>();

  if (terms.size() == 1)
    return Maybe<GatePosition>(terms[0]);

  opCount.numAdds += terms.size() - 1;
  return Maybe<GatePosition>(reduce(GateDescription::ADD, terms));
}

Maybe<GatePosition> PWSCircuitParser::
parsePolyTerm(Tokenizer& pws)
{
  vector<GatePosition> vars;
  bool abort = false;
  bool negate = false;

  //cout << "[ParseTerm] begin" << endl;
  while (pws.hasNext())
  {
    pws >> token;

    //cout << "[ParseTerm] token: " << token << endl;
    if (token == "(")
    {
      Maybe<GatePosition> gate = parsePoly(pws);
      if (!gate.isValid())
      {
        abort = true;
        break;
      }
      vars.push_back(gate.value());
    }
    else if (token == "E" || token == ")")
    {
      pws.rewind();
      break;
    }
    else if (token == "*")
    {
      if (vars.empty())
      {
        parseError("Unexpected '*' in polynomial definition.");
        abort = true;
        break;
      }
    }
    else if (token == "+")
    {
      if (vars.empty())
      {
        parseError("Unexpected '+' in polynomial definition.");
        abort = true;
      }
      else
      {
        pws.rewind();
      }
      break;
    }
    else if (token == "-")
    {
      negate = true;
      continue;
    }
    else
    {
      pws.rewind();
      Maybe<Primitive> p = parsePrimitive(pws);
      if (!p.isValid())
      {
        abort = true;
        break;
      }

      vars.push_back(getGate(p.value()));
    }

    if (negate)
    {
      vars.back() = negateGate(vars.back());
      negate = false;
    }
  }

  //cout << "[ParseTerm] end" << endl;
  if (abort || vars.empty())
    return Maybe<GatePosition>();

  if (vars.size() == 1)
    return Maybe<GatePosition>(vars[0]);

  opCount.numMults += vars.size() - 1;
  //cout << "[ParseTerm] end" << endl;
  return Maybe<GatePosition>(reduce(GateDescription::MUL, vars));
}

void PWSCircuitParser::
parsePolyConstraint(Tokenizer& pws)
{
  Maybe<Primitive> p = parsePrimitive(pws);

  if (!p.isValid())
  {
    parseError("No primitive on lhs of polynomial.");
    return;
  }

  ASSERT_RETURN(pws, "=", "polynomial definition");

  Maybe<GatePosition> gate = parsePoly(pws, "E");

  if (gate.isValid())
  {
    if (p.value().isVariable())
      bindVariable(p.value(), gate.value());
    else
      bindOutputConstant(p.value(), gate.value());
  }
}

void PWSCircuitParser::
parseLessThanInt(Tokenizer& pws)
{
  vector<int> opGuard;
  getOpGuard(opGuard);

  ASSERT_RETURN(pws, "N_0", "less than int.");
  Maybe<Primitive> n0 = parsePrimitive(pws);
  IF_INVALID_RETURN(n0);

  ASSERT_RETURN(pws, "N", "less than int.");
  int nVars;
  pws >> nVars;
  nVars--;

  ASSERT_RETURN(pws, "Mlt", "less than int.");
  Maybe<Primitive> mlt = parsePrimitive(pws);
  IF_INVALID_RETURN(mlt);

  ASSERT_RETURN(pws, "Meq", "less than int.");
  Maybe<Primitive> meq = parsePrimitive(pws);
  IF_INVALID_RETURN(meq);

  ASSERT_RETURN(pws, "Mgt", "less than int.");
  Maybe<Primitive> mgt = parsePrimitive(pws);
  IF_INVALID_RETURN(mgt);

  ASSERT_RETURN(pws, "X1", "less than int.");
  Maybe<Primitive> x1 = parsePrimitive(pws);
  IF_INVALID_RETURN(x1);

  ASSERT_RETURN(pws, "X2", "less than int.");
  Maybe<Primitive> x2 = parsePrimitive(pws);
  IF_INVALID_RETURN(x2);

  ASSERT_RETURN(pws, "Y", "less than int.");
  Maybe<Primitive> y = parsePrimitive(pws);
  IF_INVALID_RETURN(y);

  vector<Primitive> ns;
  mpz_t num;
  mpz_init_set_si(num, 1);

  vector<GatePosition> nis;
  for (int i = 0; i < nVars; i++)
  {
    CVar& ni = addVariable(CVar::MAGIC, n0.value().idx + i);
    ni.makeMagic();

    // This is contrary to the standard.
    // Constraint: Ni * (Ni + 2^i) = 0
    makeZeroOrOtherGate(ni, num);

    nis.push_back(getGate(ni));
    mpz_mul_2exp(num, num, 1);
  }
  mpz_neg(num, num);
  nis.push_back(getGate(addConstant(num)));

  makeMagic(mlt.value());
  makeMagic(meq.value());
  makeMagic(mgt.value());

  // Constraint: Mlt * (1 - Mlt) = 0
  mpz_set_ui(num, 1);
  makeZeroOrOtherGate(getCircuitPrimitive(mlt.value()), num);
  makeZeroOrOtherGate(getCircuitPrimitive(meq.value()), num);
  makeZeroOrOtherGate(getCircuitPrimitive(mgt.value()), num);

  // Constraint Mlt + Meq + Mgt = 1
  vector<GatePosition> ms;
  ms.push_back(getGate(mlt.value()));
  ms.push_back(getGate(meq.value()));
  ms.push_back(getGate(mgt.value()));
  bindOutputConstant(reduce(GateDescription::ADD, ms), 1);

  GatePosition sumNis = reduce(GateDescription::ADD, nis);
  GatePosition x1Subx2 = makeSubGate(getGate(x1.value()), getGate(x2.value()));
  GatePosition x2Subx1 = makeSubGate(getGate(x2.value()), getGate(x1.value()));

  // Constraint: Mlt * (x2 - x1 + n0 + n1 + ... - 2^N) = 0
  bindOutputConstant(
      makeGate(GateDescription::MUL,
               ms[0],
               makeGate(GateDescription::ADD, sumNis, x2Subx1)),
      0);

  // Constraint: Mgt * (x1 - x2 + n0 + n1 + ... - 2^N) = 0
  bindOutputConstant(
      makeGate(GateDescription::MUL,
               ms[2],
               makeGate(GateDescription::ADD, sumNis, x1Subx2)),
      0);

  // Constraint: Meq * (x1 - x2) = 0
  bindOutputConstant(makeGate(GateDescription::MUL, ms[1], x1Subx2), 0);

  // Bind Y = Mlt
  bindVariable(y.value(), ms[0]);

  nis.pop_back();
  addMagicOp(
      new LessThanIntOperation(
            ms,
            nis,
            getGate(x1.value()),
            getGate(x2.value())),
      opGuard);

  opCount.numCmps++;
  mpz_clear(num);
}
void PWSCircuitParser::
parseLessThanFloat(Tokenizer& pws)
{
  vector<int> opGuard;
  getOpGuard(opGuard);

  ASSERT_RETURN(pws, "N_0", "less than int.");
  Maybe<Primitive> n0 = parsePrimitive(pws);
  IF_INVALID_RETURN(n0);

  ASSERT_RETURN(pws, "Na", "less than int.");
  int nAVars;
  pws >> nAVars;

  ASSERT_RETURN(pws, "N", "less than int.");
  Maybe<Primitive> n = parsePrimitive(pws);
  IF_INVALID_RETURN(n);

  ASSERT_RETURN(pws, "D_0", "less than int.");
  Maybe<Primitive> d0 = parsePrimitive(pws);
  IF_INVALID_RETURN(d0);

  ASSERT_RETURN(pws, "Nb", "less than int.");
  int nBVars;
  pws >> nBVars;

  ASSERT_RETURN(pws, "D", "less than int.");
  Maybe<Primitive> d = parsePrimitive(pws);
  IF_INVALID_RETURN(d);

  ASSERT_RETURN(pws, "ND", "less than int.");
  Maybe<Primitive> nd = parsePrimitive(pws);
  IF_INVALID_RETURN(nd);

  ASSERT_RETURN(pws, "Mlt", "less than int.");
  Maybe<Primitive> mlt = parsePrimitive(pws);
  IF_INVALID_RETURN(mlt);

  ASSERT_RETURN(pws, "Meq", "less than int.");
  Maybe<Primitive> meq = parsePrimitive(pws);
  IF_INVALID_RETURN(meq);

  ASSERT_RETURN(pws, "Mgt", "less than int.");
  Maybe<Primitive> mgt = parsePrimitive(pws);
  IF_INVALID_RETURN(mgt);

  ASSERT_RETURN(pws, "X1", "less than int.");
  Maybe<Primitive> x1 = parsePrimitive(pws);
  IF_INVALID_RETURN(x1);

  ASSERT_RETURN(pws, "X2", "less than int.");
  Maybe<Primitive> x2 = parsePrimitive(pws);
  IF_INVALID_RETURN(x2);

  ASSERT_RETURN(pws, "Y", "less than int.");
  Maybe<Primitive> y = parsePrimitive(pws);
  IF_INVALID_RETURN(y);

  vector<Primitive> ns;
  mpz_t num, num2, inv;
  mpz_init_set_si(num, 1);
  mpz_init(num2);
  mpz_init(inv);

  vector<GatePosition> nis;
  for (int i = 0; i < nAVars; i++)
  {
    CVar& ni = addVariable(CVar::MAGIC, n0.value().idx + i);
    ni.makeMagic();

    // Constraint: Ni * (Ni - 2^i) = 0
    makeZeroOrOtherGate(ni, num);

    nis.push_back(getGate(ni));

    mpz_mul_2exp(num, num, 1);
  }
  mpz_neg(num, num);
  nis.push_back(getGate(addConstant(num)));

  // Constraint: N0 + N1 + ... - 2^Na
  GatePosition nGate = reduce(GateDescription::ADD, nis);
  bindVariable(n.value(), nGate);

  vector<GatePosition> bitDis;
  vector<GatePosition> dis;
  mpz_set_ui(num, 1);
  mpz_set_ui(num2, 1);
  mpz_set_ui(inv, 2);
  mpz_invert(inv, inv, prime);
  for (int i = 0; i < nBVars + 1; i++)
  {
    CVar& di = addVariable(CVar::MAGIC, d0.value().idx + i);
    di.makeMagic();

    // Constraint: Di * (Di - 1) = 0
    makeZeroOrOtherGate(di, num);

    GatePosition constGate = getGate(addConstant(num2));
    GatePosition diGate = getGate(di);
    GatePosition diProdConstGate = makeGate(GateDescription::MUL, constGate, diGate);

    bitDis.push_back(diGate);
    dis.push_back(diProdConstGate);

    modmult(num2, num2, inv, prime);
  }

  // Constraint: D0 + D1 + ... + DNa = 1
  GatePosition diUniqueGate = reduce(GateDescription::ADD, bitDis);
  bindOutputConstant(diUniqueGate, 1);

  // Constraint: D0 + D1/2 + ... + Di/2^i = D
  GatePosition dGate = reduce(GateDescription::ADD, dis);
  bindVariable(d.value(), dGate);

  // Constraint: ND = N * D
  GatePosition ndGate = makeGate(GateDescription::MUL, nGate, dGate);
  bindVariable(nd.value(), ndGate);

  makeMagic(mlt.value());
  makeMagic(meq.value());
  makeMagic(mgt.value());

  // Constraint: Mlt * (1 - Mlt) = 0
  mpz_set_ui(num, 1);
  makeZeroOrOtherGate(getCircuitPrimitive(mlt.value()), num);
  makeZeroOrOtherGate(getCircuitPrimitive(meq.value()), num);
  makeZeroOrOtherGate(getCircuitPrimitive(mgt.value()), num);

  // Constraint Mlt + Meq + Mgt = 1
  vector<GatePosition> ms;
  ms.push_back(getGate(mlt.value()));
  ms.push_back(getGate(meq.value()));
  ms.push_back(getGate(mgt.value()));
  bindOutputConstant(reduce(GateDescription::ADD, ms), 1);

  GatePosition x1Subx2 = makeSubGate(getGate(x1.value()), getGate(x2.value()));
  GatePosition x2Subx1 = makeSubGate(getGate(x2.value()), getGate(x1.value()));

  // Constraint: Mlt * (x2 - x1 + ND) = 0
  bindOutputConstant(
      makeGate(GateDescription::MUL,
               ms[0],
               makeGate(GateDescription::ADD, ndGate, x2Subx1)),
      0);

  // Constraint: Mgt * (x1 - x2 + ND) = 0
  bindOutputConstant(
      makeGate(GateDescription::MUL,
               ms[2],
               makeGate(GateDescription::ADD, ndGate, x1Subx2)),
      0);

  // Constraint: Meq * (x1 - x2) = 0
  bindOutputConstant(makeGate(GateDescription::MUL, ms[1], x1Subx2), 0);

  // Bind Y = Mlt
  bindVariable(y.value(), ms[0]);

  nis.pop_back();
  addMagicOp(
      new LessThanFloatOperation(
            ms,
            nis,
            bitDis,
            getGate(x1.value()),
            getGate(x2.value())),
      opGuard);

  opCount.numCmps++;

  mpz_clear(num);
  mpz_clear(num2);
  mpz_clear(inv);
}

void PWSCircuitParser::
parseNotEqual(Tokenizer& pws)
{
  // Record opguard first.
  vector<int> opGuard;
  getOpGuard(opGuard);

  ASSERT_RETURN(pws, "M", "less than int.");
  Maybe<Primitive> m = parsePrimitive(pws);
  IF_INVALID_RETURN(m);

  ASSERT_RETURN(pws, "X1", "less than int.");
  Maybe<Primitive> x1 = parsePrimitive(pws);
  IF_INVALID_RETURN(x1);

  ASSERT_RETURN(pws, "X2", "less than int.");
  Maybe<Primitive> x2 = parsePrimitive(pws);
  IF_INVALID_RETURN(x2);

  ASSERT_RETURN(pws, "Y", "less than int.");
  Maybe<Primitive> y = parsePrimitive(pws);
  IF_INVALID_RETURN(y);

  makeMagic(m.value());

  GatePosition mGate = getGate(m.value());
  GatePosition x1Subx2 = makeSubGate(getGate(x1.value()), getGate(x2.value()));
  GatePosition x2Subx1 = makeSubGate(getGate(x2.value()), getGate(x1.value()));

  // Constraint: M * (X1 - X2) = Y.
  GatePosition mTimesDiff = makeGate(GateDescription::MUL, mGate, x1Subx2);

  // Gate: -(X1 - X2) * Y.
  GatePosition negMTimesDiff = makeGate(GateDescription::MUL, mTimesDiff, x2Subx1);

  // Constraint: (X1 - X2) - (X1 - X2) * Y = 0
  GatePosition check = makeGate(GateDescription::ADD, negMTimesDiff, x1Subx2);

  bindVariable(y.value(), mTimesDiff);
  bindOutputConstant(check, 0);

  addMagicOp(
      new NotEqualOperation(
            mGate,
            getGate(x1.value()),
            getGate(x2.value())),
      opGuard);

  opCount.numIneqs++;
}

void PWSCircuitParser::
parseDivide(Tokenizer& pws)
{
  Maybe<Primitive> p = parsePrimitive(pws);

  if (!p.isValid() || !p.value().isVariable())
  {
    parseError("No named variable on lhs of polynomial.");
    return;
  }

  ASSERT_RETURN(pws, "=", "divide constraint.");
  pws >> token;
  Maybe<Primitive> dividend = parseVar(token);
  IF_INVALID_RETURN(dividend);

  ASSERT_RETURN(pws, "/", "divide constraint.");
  pws >> token;

  mpz_t divisor;
  mpz_init(divisor);

  if (mpz_set_str(divisor, token.c_str(), 10) == 0)
  {
    mpz_invert(divisor, divisor, prime);
    GatePosition divisorGate = getGate(addConstant(divisor));
    GatePosition dividendGate = getGate(dividend.value());

    // This is an ugly hack to make this all work. We create a DIV_INT gate,
    // which functions as a MUL gate, but with the additional information that
    // the second operand is actually an integer to divide by.
    GatePosition quotientGate = makeGate(GateDescription::DIV_INT, dividendGate, divisorGate);
    bindVariable(p.value(), quotientGate);
  }
  else
  {
    parseError("No Divisor found.");
  }

  opCount.numIntDivs++;
  mpz_clear(divisor);
}

void PWSCircuitParser::
printCircuitDescription()
{
  cout << "=== CONSTANTS ===" << endl;
  typedef vector<CConst*>::const_iterator CConstIt;
  for (CConstIt it = constants.begin(); it != constants.end(); ++it)
  {
    if ((*it)->isBound())
      printf("%02d || %s\n", (*it)->gateIndex.front(), (*it)->value.c_str());
  }

  cout << endl << "=== IO ===" << endl;
  for (CVarIt it = inOutVarsMap.begin(); it != inOutVarsMap.end(); ++it)
  {
    CVar* cVar = it->second;
    if (!cVar->isBound())
      continue;

    if (cVar->isInput())
    {
      printf("%02d || %s%d\n", cVar->gateIndex.front(),
                               cVar->varTypeStr().c_str(),
                               cVar->name);
    }
    else
    {
      printf("%02d || %s%d\n", cVar->gateIndex.back(),
                               cVar->varTypeStr().c_str(),
                               cVar->name);
    }
  }

  cout << endl << "=== VARIABLES ===" << endl;
  for (CVarIt it = varMap.begin(); it != varMap.end(); ++it)
  {
    CVar* cVar = it->second;
    printf("%02d || %s%d\n", cVar->gateIndex.front(),
                             cVar->varTypeStr().c_str(),
                             cVar->name);
  }

  cout << endl << "=== OUT CONSTS ===" << endl;
  for (size_t i = 0; i < outConstantDesc.size(); i++)
  {
    cout << outConstantDesc[i].first << ": " << outConstantDesc[i].second << endl;
  }

  cout << endl;
  for (size_t i = 0; i < circuitDesc.size(); i++)
  {
    LayerDescription& layer = circuitDesc[i];
    for (size_t j = 0; j < layer.size(); j++)
    {
      GateDescription& gate = layer[j];
      printf("[%d] %s || %02d || %02d | %02d",
             i,
             gate.strOpType().c_str(),
             gate.pos.name,
             gate.in1,
             gate.in2);

      if (gate.isBound())
      {
        cout << " || " << getCircuitPrimitive(gate.boundPrimitive.value()) << endl;
      }
      else
      {
        cout << endl;
      }
    }
    cout << endl;
  }
}

template<class T> void
printMemoryStat(const string name, const T& container, size_t nLayers)
{
  size_t nelems = container.size();
  size_t nbytes = nelems * sizeof(typename T::const_iterator::value_type) + sizeof(T);

  cout << name << nelems << " | " << (nbytes / 1024.0 / 1024.0) << " Mbytes." << endl;
}

template<> void
printMemoryStat(const string name, const vector<CConst*>& container, size_t nLayers)
{
  const size_t avgNumChars = 10;
  size_t nelems = container.size();
  size_t nbytes = sizeof(vector<CConst*>) +
                  nelems * (sizeof(CConst*) + sizeof(CConst) + nLayers * sizeof(int) + avgNumChars);

  cout << name << nelems << " | " << (nbytes / 1024.0 / 1024.0) << " Mbytes." << endl;
}


void PWSCircuitParser::
printMemoryStats() const
{
  size_t nLayers = circuitDesc.size();
  printMemoryStat("varMapsize:       ", varMap, nLayers);
  printMemoryStat("inoutvarsmapsize: ", inOutVarsMap, nLayers);
  printMemoryStat("constsize:        ", constants, nLayers);
  printMemoryStat("constmapsize:     ", constMap, nLayers);
  printMemoryStat("outconstdescsize: ", outConstantDesc, nLayers);
  printMemoryStat("circuitdescsize:  ", circuitDesc, nLayers);
  printMemoryStat("magicopssize:     ", magicOps, nLayers);

  /*
  cout<< "varMapsize:       " << varMap.size() << endl;;
  cout<< "inoutvarsmapsize: " << inOutVarsMap.size() << endl;;
  cout<< "constsize:        " << constants.size() << endl;;
  cout<< "constmapsize:     " << constMap.size() << endl;;
  cout<< "outconstdescsize: " << outConstantDesc.size() << endl;;
  cout<< "circuitdescsize:  " << circuitDesc.size() << endl;;
  cout<< "magicopssize:     " << magicOps.size() << endl;;
  */
}

