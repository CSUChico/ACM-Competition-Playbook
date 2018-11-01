
## Algorithms:

### BellmanFord.cc
```cpp
// This function runs the Bellman-Ford algorithm for single source
// shortest paths with negative edge weights.  The function returns
// false if a negative weight cycle is detected.  Otherwise, the
// function returns true and dist[i] is the length of the shortest
// path from start to i.
//
// Edge case: watch out for negative cycles.
// 
// Running time: O(|V|^3)
//
//   INPUT:   start, w[i][j] = cost of edge from i to j
//   OUTPUT:  dist[i] = min weight path from start to i
//            prev[i] = previous node on the best path from the
//                      start node   

#include <iostream>
#include <queue>
#include <cmath>
#include <vector>

using namespace std;

typedef double T;
typedef vector<T> VT;
typedef vector<VT> VVT;

typedef vector<int> VI;
typedef vector<VI> VVI;

bool BellmanFord (const VVT &w, VT &dist, VI &prev, int start){
  int n = w.size();
  prev = VI(n, -1);
  dist = VT(n, 1000000000);
  dist[start] = 0;
  
  for (int k = 0; k < n; k++){
    for (int i = 0; i < n; i++){
      for (int j = 0; j < n; j++){
        if (dist[j] > dist[i] + w[i][j]){
          if (k == n-1) return false;
          dist[j] = dist[i] + w[i][j];
          prev[j] = i;
        }	  
      }
    }
  }
  
  return true;
}
```

### Breadth First Search(BFS)
```cpp  
#include <list>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

vector<list<int>> adjacencyList;
vector<int> dist;
vector<int> parent;

void BFS(int start)
{
  vector<char> color(adjacencyList.size());
  for(unsigned int i = 0; i < adjacencyList.size(); i++)
  {
    color[i] = 'w';
    dist[i] = INT_MAX;
    parent[i] = -1;
  }
  color[start] = 'g';
  dist[start] = 0;
  queue<int> Q;
  Q.push(start);
  int current;
  while(!Q.empty())
  {
    current = Q.front();
    Q.pop();
    for(int i : adjacencyList[current])
    {
      if(color[i] == 'w')
      {
        color[i] = 'g';
        dist[i] = dist[current] + 1;
        parent[i] = current;
        Q.push(i);
      }
    }
    color[current] = 'b';
  }
}
```

### CSP.cc
```cpp
// Constraint satisfaction problems

#include <cstdlib>
#include <iostream>
#include <vector>
#include <set>
using namespace std;

#define DONE   -1
#define FAILED -2

typedef vector<int> VI;
typedef vector<VI> VVI;
typedef vector<VVI> VVVI;

typedef set<int> SI;

// Lists of assigned/unassigned variables.
VI assigned_vars;
SI unassigned_vars;

// For each variable, a list of reductions (each of which a list of eliminated
// variables)
VVVI reductions;

// For each variable, a list of the variables whose domains it reduced in
// forward-checking.
VVI forward_mods;

// need to implement ----------------------------
int Value(int var);

void SetValue(int var, int value);
void ClearValue(int var);

int DomainSize(int var);
void ResetDomain(int var);
void AddValue(int var, int value);
void RemoveValue(int var, int value);

int NextVar() {
  if ( unassigned_vars.empty() ) return DONE;
  
  // could also do most constrained...
  int var = *unassigned_vars.begin();
  return var;
}

int Initialize() {
  // setup here
  return NextVar();
}
// ------------------------- end -- need to implement


void UpdateCurrentDomain(int var) {
  ResetDomain(var);
  for (int i = 0; i < reductions[var].size(); i++) {
    vector<int>& red = reductions[var][i];
    for (int j = 0; j < red.size(); j++) {
      RemoveValue(var, red[j]);
    }
  }
}


void UndoReductions(int var) {
  for (int i = 0; i < forward_mods[var].size(); i++) {
    int other_var = forward_mods[var][i];
    VI& red = reductions[other_var].back();
    for (int j = 0; j < red.size(); j++) {
      AddValue(other_var, red[j]);
    }
    reductions[other_var].pop_back();
  }
  forward_mods[var].clear();
}


bool ForwardCheck(int var, int other_var) {
  vector<int> red;
  
  foreach value in current_domain(other_var) {
    SetValue(other_var, value);
    if ( !Consistent(var, other_var) ) {
      red.push_back(value);
      RemoveValue(other_var, value);
    }
    ClearValue(other_var);
  }
  if ( !red.empty() ) {
    reductions[other_var].push_back(red);
    forward_mods[var].push_back(other_var);
  }
  
  return DomainSize(other_var) != 0;
}


pair<int, bool> Unlabel(int var) {
  assigned_vars.pop_back();
  unassigned_vars.insert(var);
  
  UndoReductions(var);
  UpdateCurrentDomain(var);
  
  if ( assigned_vars.empty() ) return make_pair(FAILED, true);
  
  int prev_var = assigned_vars.back();
  RemoveValue(prev_var, Value(prev_var));
  ClearValue(prev_var);
  if ( DomainSize(prev_var) == 0 ) {
    return make_pair(prev_var, false);
  } else {
    return make_pair(prev_var, true);
  }
}


pair<int, bool> Label(int var) {
  unassigned_vars.erase(var);
  assigned_vars.push_back(var);
  
  bool consistent;
  foreach value in current_domain(var) {
    SetValue(var, value);
    consistent = true;
    for (int j=0; j<unassigned_vars.size(); j++) {
      int other_var = unassigned_vars[j];
      if ( !ForwardCheck(var, other_var) ) {
        RemoveValue(var, value);
        consistent = false;
        UndoReductions(var);
        ClearValue(var);
        break;
      }
    }
    if ( consistent ) return (NextVar(), true);
  }
  return make_pair(var, false);
}


void BacktrackSearch(int num_var) {
  // (next variable to mess with, whether current state is consistent)
  pair<int, bool> var_consistent = make_pair(Initialize(), true);
  while ( true ) {
    if ( var_consistent.second ) var_consistent = Label(var_consistent.first);
    else var_consistent = Unlabel(var_consistent.first);
    
    if ( var_consistent.first == DONE ) return; // solution found
    if ( var_consistent.first == FAILED ) return; // no solution
  }
}
```
### ConvexHull.cc
```cpp
// Compute the 2D convex hull of a set of points using the monotone chain
// algorithm.  Eliminate redundant points from the hull if REMOVE_REDUNDANT is 
// #defined.
//
// Running time: O(n log n)
//
//   INPUT:   a vector of input points, unordered.
//   OUTPUT:  a vector of points in the convex hull, counterclockwise, starting
//            with bottommost/leftmost point

#include <cstdio>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cmath>
// BEGIN CUT
#include <map>
// END CUT

using namespace std;

#define REMOVE_REDUNDANT

typedef double T;
const T EPS = 1e-7;
struct PT { 
  T x, y; 
  PT() {} 
  PT(T x, T y) : x(x), y(y) {}
  bool operator<(const PT &rhs) const { return make_pair(y,x) < make_pair(rhs.y,rhs.x); }
  bool operator==(const PT &rhs) const { return make_pair(y,x) == make_pair(rhs.y,rhs.x); }
};

T cross(PT p, PT q) { return p.x*q.y-p.y*q.x; }
T area2(PT a, PT b, PT c) { return cross(a,b) + cross(b,c) + cross(c,a); }

#ifdef REMOVE_REDUNDANT
bool between(const PT &a, const PT &b, const PT &c) {
  return (fabs(area2(a,b,c)) < EPS && (a.x-b.x)*(c.x-b.x) <= 0 && (a.y-b.y)*(c.y-b.y) <= 0);
}
#endif

void ConvexHull(vector<PT> &pts) {
  sort(pts.begin(), pts.end());
  pts.erase(unique(pts.begin(), pts.end()), pts.end());
  vector<PT> up, dn;
  for (int i = 0; i < pts.size(); i++) {
    while (up.size() > 1 && area2(up[up.size()-2], up.back(), pts[i]) >= 0) up.pop_back();
    while (dn.size() > 1 && area2(dn[dn.size()-2], dn.back(), pts[i]) <= 0) dn.pop_back();
    up.push_back(pts[i]);
    dn.push_back(pts[i]);
  }
  pts = dn;
  for (int i = (int) up.size() - 2; i >= 1; i--) pts.push_back(up[i]);
  
#ifdef REMOVE_REDUNDANT
  if (pts.size() <= 2) return;
  dn.clear();
  dn.push_back(pts[0]);
  dn.push_back(pts[1]);
  for (int i = 2; i < pts.size(); i++) {
    if (between(dn[dn.size()-2], dn[dn.size()-1], pts[i])) dn.pop_back();
    dn.push_back(pts[i]);
  }
  if (dn.size() >= 3 && between(dn.back(), dn[0], dn[1])) {
    dn[0] = dn.back();
    dn.pop_back();
  }
  pts = dn;
#endif
}

// BEGIN CUT
// The following code solves SPOJ problem #26: Build the Fence (BSHEEP)

int main() {
  int t;
  scanf("%d", &t);
  for (int caseno = 0; caseno < t; caseno++) {
    int n;
    scanf("%d", &n);
    vector<PT> v(n);
    for (int i = 0; i < n; i++) scanf("%lf%lf", &v[i].x, &v[i].y);
    vector<PT> h(v);
    map<PT,int> index;
    for (int i = n-1; i >= 0; i--) index[v[i]] = i+1;
    ConvexHull(h);
    
    double len = 0;
    for (int i = 0; i < h.size(); i++) {
      double dx = h[i].x - h[(i+1)%h.size()].x;
      double dy = h[i].y - h[(i+1)%h.size()].y;
      len += sqrt(dx*dx+dy*dy);
    }
    
    if (caseno > 0) printf("\n");
    printf("%.2f\n", len);
    for (int i = 0; i < h.size(); i++) {
      if (i > 0) printf(" ");
      printf("%d", index[h[i]]);
    }
    printf("\n");
  }
}

// END CUT
```

### Dates.cc
```cpp
// Routines for performing computations on dates.  In these routines,
// months are expressed as integers from 1 to 12, days are expressed
// as integers from 1 to 31, and years are expressed as 4-digit
// integers.

#include <iostream>
#include <string>

using namespace std;

string dayOfWeek[] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};

// converts Gregorian date to integer (Julian day number)
int dateToInt (int m, int d, int y){  
  return 
    1461 * (y + 4800 + (m - 14) / 12) / 4 +
    367 * (m - 2 - (m - 14) / 12 * 12) / 12 - 
    3 * ((y + 4900 + (m - 14) / 12) / 100) / 4 + 
    d - 32075;
}

// converts integer (Julian day number) to Gregorian date: month/day/year
void intToDate (int jd, int &m, int &d, int &y){
  int x, n, i, j;
  
  x = jd + 68569;
  n = 4 * x / 146097;
  x -= (146097 * n + 3) / 4;
  i = (4000 * (x + 1)) / 1461001;
  x -= 1461 * i / 4 - 31;
  j = 80 * x / 2447;
  d = x - 2447 * j / 80;
  x = j / 11;
  m = j + 2 - 12 * x;
  y = 100 * (n - 49) + i + x;
}

// converts integer (Julian day number) to day of week
string intToDay (int jd){
  return dayOfWeek[jd % 7];
}

int main (int argc, char **argv){
  int jd = dateToInt (3, 24, 2004);
  int m, d, y;
  intToDate (jd, m, d, y);
  string day = intToDay (jd);
  
  // expected output:
  //    2453089
  //    3/24/2004
  //    Wed
  cout << jd << endl
    << m << "/" << d << "/" << y << endl
    << day << endl;
}
```

### Delaunay.cc
```cpp
// Slow but simple Delaunay triangulation. Does not handle
// degenerate cases (from O'Rourke, Computational Geometry in C)
//
// Running time: O(n^4)
//
// INPUT:    x[] = x-coordinates
//           y[] = y-coordinates
//
// OUTPUT:   triples = a vector containing m triples of indices
//                     corresponding to triangle vertices

#include<vector>
using namespace std;

typedef double T;

struct triple {
    int i, j, k;
    triple() {}
    triple(int i, int j, int k) : i(i), j(j), k(k) {}
};

vector<triple> delaunayTriangulation(vector<T>& x, vector<T>& y) {
	int n = x.size();
	vector<T> z(n);
	vector<triple> ret;

	for (int i = 0; i < n; i++)
	    z[i] = x[i] * x[i] + y[i] * y[i];

	for (int i = 0; i < n-2; i++) {
	    for (int j = i+1; j < n; j++) {
		for (int k = i+1; k < n; k++) {
		    if (j == k) continue;
		    double xn = (y[j]-y[i])*(z[k]-z[i]) - (y[k]-y[i])*(z[j]-z[i]);
		    double yn = (x[k]-x[i])*(z[j]-z[i]) - (x[j]-x[i])*(z[k]-z[i]);
		    double zn = (x[j]-x[i])*(y[k]-y[i]) - (x[k]-x[i])*(y[j]-y[i]);
		    bool flag = zn < 0;
		    for (int m = 0; flag && m < n; m++)
			flag = flag && ((x[m]-x[i])*xn + 
					(y[m]-y[i])*yn + 
					(z[m]-z[i])*zn <= 0);
		    if (flag) ret.push_back(triple(i, j, k));
		}
	    }
	}
	return ret;
}

int main()
{
    T xs[]={0, 0, 1, 0.9};
    T ys[]={0, 1, 0, 0.9};
    vector<T> x(&xs[0], &xs[4]), y(&ys[0], &ys[4]);
    vector<triple> tri = delaunayTriangulation(x, y);
    
    //expected: 0 1 3
    //          0 3 2
    
    int i;
    for(i = 0; i < tri.size(); i++)
        printf("%d %d %d\n", tri[i].i, tri[i].j, tri[i].k);
    return 0;
}
```

### Depth First Search(DFS)
```cpp  
#include <vector>
#include <list>

using namespace std;

vector<list<int>> adjacencyList;
vector<int> parent;
vector<int> startTime;
vector<int> finishTime;
vector<char> color;


void DFS_Visit(int current, int & time)
{
  color[current] = 'g';
  time++;
  startTime[current] = time;
  for(int i : adjacencyList[current])
  {
    if(color[i] == 'w')
    {
      parent[i] = current;
      DFS_Visit(i, time);
    }
  }
  color[current] = 'b';
  time++;
  finishTime[current] = time;
}

void DFS()
{
  for(int i = 0; i < adjacencyList.size(); i++)
  {
    color[i] = 'w';
    parent[i] = -1;
  }
  int time = 0;
  for(int i = 0; i < adjacencyList.size(); i++)
  {
    if(color[i] == 'w')
    {
      DFS_Visit(i, time);
    }
  }
}
```

### Dijkstra
```cpp
// Single Source Shortest Path
// Inputs:
//   AdjacencyList
//   Start

#include <queue>
#include <vector>
#include <list>
#include <climits>

using namespace std;

struct edge
{
  int to;
  int weight;
};

struct node
{
  int index;
  int cost;
};

struct compare
{
  bool operator() (node a, node b)
  {
    return a.cost > b.cost;
  }
};

vector<list<edge>> adjacencyList;
vector<int> dist;
vector<int> parent;

void Dijkstra(int start)
{
  priority_queue<node, vector<node>, compare> Q;
  for(int & i : dist)
  {
    i = INT_MAX;
  }
  node source;
  source.index = start;
  source.cost = 0;
  dist[start] = 0;
  parent[start] = -1;
  Q.push(source);
  vector<bool> finished(adjacencyList.size(), false);

  node current;
  while(!Q.empty())
  {
    current = Q.top();
    Q.pop();
    node temp;
    for(edge i : adjacencyList[current.index])
    {
      if(!finished[i.to] && dist[i.to] > dist[current.index] + i.weight)
      {
        dist[i.to] = dist[current.index] + i.weight;
        temp.index = i.to;
        temp.cost = dist[i.to];
        parent[i.to] = current.index;
        Q.push(temp);
      }
    }
    finished[current.index] = true;
  }
}
```

### Dinic.cc
```cpp
// Adjacency list implementation of Dinic's blocking flow algorithm.
// This is very fast in practice, and only loses to push-relabel flow.
//
// Running time:
//     O(|V|^2 |E|)
//
// INPUT:
//     - graph, constructed using AddEdge()
//     - source and sink
//
// OUTPUT:
//     - maximum flow value
//     - To obtain actual flow values, look at edges with capacity > 0
//       (zero capacity edges are residual edges).

#include<cstdio>
#include<vector>
#include<queue>
using namespace std;
typedef long long LL;

struct Edge {
  int u, v;
  LL cap, flow;
  Edge() {}
  Edge(int u, int v, LL cap): u(u), v(v), cap(cap), flow(0) {}
};

struct Dinic {
  int N;
  vector<Edge> E;
  vector<vector<int>> g;
  vector<int> d, pt;
  
  Dinic(int N): N(N), E(0), g(N), d(N), pt(N) {}

  void AddEdge(int u, int v, LL cap) {
    if (u != v) {
      E.emplace_back(Edge(u, v, cap));
      g[u].emplace_back(E.size() - 1);
      E.emplace_back(Edge(v, u, 0));
      g[v].emplace_back(E.size() - 1);
    }
  }

  bool BFS(int S, int T) {
    queue<int> q({S});
    fill(d.begin(), d.end(), N + 1);
    d[S] = 0;
    while(!q.empty()) {
      int u = q.front(); q.pop();
      if (u == T) break;
      for (int k: g[u]) {
        Edge &e = E[k];
        if (e.flow < e.cap && d[e.v] > d[e.u] + 1) {
          d[e.v] = d[e.u] + 1;
          q.emplace(e.v);
        }
      }
    }
    return d[T] != N + 1;
  }

  LL DFS(int u, int T, LL flow = -1) {
    if (u == T || flow == 0) return flow;
    for (int &i = pt[u]; i < g[u].size(); ++i) {
      Edge &e = E[g[u][i]];
      Edge &oe = E[g[u][i]^1];
      if (d[e.v] == d[e.u] + 1) {
        LL amt = e.cap - e.flow;
        if (flow != -1 && amt > flow) amt = flow;
        if (LL pushed = DFS(e.v, T, amt)) {
          e.flow += pushed;
          oe.flow -= pushed;
          return pushed;
        }
      }
    }
    return 0;
  }

  LL MaxFlow(int S, int T) {
    LL total = 0;
    while (BFS(S, T)) {
      fill(pt.begin(), pt.end(), 0);
      while (LL flow = DFS(S, T))
        total += flow;
    }
    return total;
  }
};

// BEGIN CUT
// The following code solves SPOJ problem #4110: Fast Maximum Flow (FASTFLOW)

int main()
{
  int N, E;
  scanf("%d%d", &N, &E);
  Dinic dinic(N);
  for(int i = 0; i < E; i++)
  {
    int u, v;
    LL cap;
    scanf("%d%d%lld", &u, &v, &cap);
    dinic.AddEdge(u - 1, v - 1, cap);
    dinic.AddEdge(v - 1, u - 1, cap);
  }
  printf("%lld\n", dinic.MaxFlow(0, N - 1));
  return 0;
}

// END CUT
```

### Euclid
```cpp
// This is a collection of useful code for solving problems that
// involve modular linear equations.  Note that all of the
// algorithms described here work on nonnegative integers.

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef vector<int> VI;
typedef pair<int, int> PII;

// return a % b (positive value)
int mod(int a, int b) {
	return ((a%b) + b) % b;
}

// computes gcd(a,b)
int gcd(int a, int b) {
	while (b) { int t = a%b; a = b; b = t; }
	return a;
}

// computes lcm(a,b)
int lcm(int a, int b) {
	return a / gcd(a, b)*b;
}

// (a^b) mod m via successive squaring
int powermod(int a, int b, int m)
{
	int ret = 1;
	while (b)
	{
		if (b & 1) ret = mod(ret*a, m);
		a = mod(a*a, m);
		b >>= 1;
	}
	return ret;
}

// returns g = gcd(a, b); finds x, y such that d = ax + by
int extended_euclid(int a, int b, int &x, int &y) {
	int xx = y = 0;
	int yy = x = 1;
	while (b) {
		int q = a / b;
		int t = b; b = a%b; a = t;
		t = xx; xx = x - q*xx; x = t;
		t = yy; yy = y - q*yy; y = t;
	}
	return a;
}

// finds all solutions to ax = b (mod n)
VI modular_linear_equation_solver(int a, int b, int n) {
	int x, y;
	VI ret;
	int g = extended_euclid(a, n, x, y);
	if (!(b%g)) {
		x = mod(x*(b / g), n);
		for (int i = 0; i < g; i++)
			ret.push_back(mod(x + i*(n / g), n));
	}
	return ret;
}

// computes b such that ab = 1 (mod n), returns -1 on failure
int mod_inverse(int a, int n) {
	int x, y;
	int g = extended_euclid(a, n, x, y);
	if (g > 1) return -1;
	return mod(x, n);
}

// Chinese remainder theorem (special case): find z such that
// z % m1 = r1, z % m2 = r2.  Here, z is unique modulo M = lcm(m1, m2).
// Return (z, M).  On failure, M = -1.
PII chinese_remainder_theorem(int m1, int r1, int m2, int r2) {
	int s, t;
	int g = extended_euclid(m1, m2, s, t);
	if (r1%g != r2%g) return make_pair(0, -1);
	return make_pair(mod(s*r2*m1 + t*r1*m2, m1*m2) / g, m1*m2 / g);
}

// Chinese remainder theorem: find z such that
// z % m[i] = r[i] for all i.  Note that the solution is
// unique modulo M = lcm_i (m[i]).  Return (z, M). On 
// failure, M = -1. Note that we do not require the a[i]'s
// to be relatively prime.
PII chinese_remainder_theorem(const VI &m, const VI &r) {
	PII ret = make_pair(r[0], m[0]);
	for (int i = 1; i < m.size(); i++) {
		ret = chinese_remainder_theorem(ret.second, ret.first, m[i], r[i]);
		if (ret.second == -1) break;
	}
	return ret;
}

// computes x and y such that ax + by = c
// returns whether the solution exists
bool linear_diophantine(int a, int b, int c, int &x, int &y) {
	if (!a && !b)
	{
		if (c) return false;
		x = 0; y = 0;
		return true;
	}
	if (!a)
	{
		if (c % b) return false;
		x = 0; y = c / b;
		return true;
	}
	if (!b)
	{
		if (c % a) return false;
		x = c / a; y = 0;
		return true;
	}
	int g = gcd(a, b);
	if (c % g) return false;
	x = c / g * mod_inverse(a / g, b / g);
	y = (c - a*x) / b;
	return true;
}

int main() {
	// expected: 2
	cout << gcd(14, 30) << endl;

	// expected: 2 -2 1
	int x, y;
	int g = extended_euclid(14, 30, x, y);
	cout << g << " " << x << " " << y << endl;

	// expected: 95 451
	VI sols = modular_linear_equation_solver(14, 30, 100);
	for (int i = 0; i < sols.size(); i++) cout << sols[i] << " ";
	cout << endl;

	// expected: 8
	cout << mod_inverse(8, 9) << endl;

	// expected: 23 105
	//           11 12
	PII ret = chinese_remainder_theorem(VI({ 3, 5, 7 }), VI({ 2, 3, 2 }));
	cout << ret.first << " " << ret.second << endl;
	ret = chinese_remainder_theorem(VI({ 4, 6 }), VI({ 3, 5 }));
	cout << ret.first << " " << ret.second << endl;

	// expected: 5 -15
	if (!linear_diophantine(7, 2, 5, x, y)) cout << "ERROR" << endl;
	cout << x << " " << y << endl;
	return 0;
}
```
### EulerianPath.cc
```cpp
struct Edge;
typedef list<Edge>::iterator iter;

struct Edge
{
	int next_vertex;
	iter reverse_edge;

	Edge(int next_vertex)
		:next_vertex(next_vertex)
		{ }
};

const int max_vertices = ;
int num_vertices;
list<Edge> adj[max_vertices];		// adjacency list

vector<int> path;

void find_path(int v)
{
	while(adj[v].size() > 0)
	{
		int vn = adj[v].front().next_vertex;
		adj[vn].erase(adj[v].front().reverse_edge);
		adj[v].pop_front();
		find_path(vn);
	}
	path.push_back(v);
}

void add_edge(int a, int b)
{
	adj[a].push_front(Edge(b));
	iter ita = adj[a].begin();
	adj[b].push_front(Edge(a));
	iter itb = adj[b].begin();
	ita->reverse_edge = itb;
	itb->reverse_edge = ita;
}
```

### FFT.cc
```cpp
// Convolution using the fast Fourier transform (FFT).
//
// INPUT:
//     a[1...n]
//     b[1...m]
//
// OUTPUT:
//     c[1...n+m-1] such that c[k] = sum_{i=0}^k a[i] b[k-i]
//
// Alternatively, you can use the DFT() routine directly, which will
// zero-pad your input to the next largest power of 2 and compute the
// DFT or inverse DFT.

#include <iostream>
#include <vector>
#include <complex>

using namespace std;

typedef long double DOUBLE;
typedef complex<DOUBLE> COMPLEX;
typedef vector<DOUBLE> VD;
typedef vector<COMPLEX> VC;

struct FFT {
  VC A;
  int n, L;

  int ReverseBits(int k) {
    int ret = 0;
    for (int i = 0; i < L; i++) {
      ret = (ret << 1) | (k & 1);
      k >>= 1;
    }
    return ret;
  }

  void BitReverseCopy(VC a) {
    for (n = 1, L = 0; n < a.size(); n <<= 1, L++) ;
    A.resize(n);
    for (int k = 0; k < n; k++) 
      A[ReverseBits(k)] = a[k];
  }
  
  VC DFT(VC a, bool inverse) {
    BitReverseCopy(a);
    for (int s = 1; s <= L; s++) {
      int m = 1 << s;
      COMPLEX wm = exp(COMPLEX(0, 2.0 * M_PI / m));
      if (inverse) wm = COMPLEX(1, 0) / wm;
      for (int k = 0; k < n; k += m) {
	COMPLEX w = 1;
	for (int j = 0; j < m/2; j++) {
	  COMPLEX t = w * A[k + j + m/2];
	  COMPLEX u = A[k + j];
	  A[k + j] = u + t;
	  A[k + j + m/2] = u - t;
	  w = w * wm;
	}
      }
    }
    if (inverse) for (int i = 0; i < n; i++) A[i] /= n;
    return A;
  }

  // c[k] = sum_{i=0}^k a[i] b[k-i]
  VD Convolution(VD a, VD b) {
    int L = 1;
    while ((1 << L) < a.size()) L++;
    while ((1 << L) < b.size()) L++;
    int n = 1 << (L+1);

    VC aa, bb;
    for (size_t i = 0; i < n; i++) aa.push_back(i < a.size() ? COMPLEX(a[i], 0) : 0);
    for (size_t i = 0; i < n; i++) bb.push_back(i < b.size() ? COMPLEX(b[i], 0) : 0);
    
    VC AA = DFT(aa, false);
    VC BB = DFT(bb, false);
    VC CC;
    for (size_t i = 0; i < AA.size(); i++) CC.push_back(AA[i] * BB[i]);
    VC cc = DFT(CC, true);

    VD c;
    for (int i = 0; i < a.size() + b.size() - 1; i++) c.push_back(cc[i].real());
    return c;
  }

};

int main() {
  double a[] = {1, 3, 4, 5, 7};
  double b[] = {2, 4, 6};

  FFT fft;
  VD c = fft.Convolution(VD(a, a + 5), VD(b, b + 3));

  // expected output: 2 10 26 44 58 58 42
  for (int i = 0; i < c.size(); i++) cerr << c[i] << " ";
  cerr << endl;
  
  return 0;
}
```
### FFT_new.cc
```cpp
#include <cassert>
#include <cstdio>
#include <cmath>

struct cpx
{
  cpx(){}
  cpx(double aa):a(aa),b(0){}
  cpx(double aa, double bb):a(aa),b(bb){}
  double a;
  double b;
  double modsq(void) const
  {
    return a * a + b * b;
  }
  cpx bar(void) const
  {
    return cpx(a, -b);
  }
};

cpx operator +(cpx a, cpx b)
{
  return cpx(a.a + b.a, a.b + b.b);
}

cpx operator *(cpx a, cpx b)
{
  return cpx(a.a * b.a - a.b * b.b, a.a * b.b + a.b * b.a);
}

cpx operator /(cpx a, cpx b)
{
  cpx r = a * b.bar();
  return cpx(r.a / b.modsq(), r.b / b.modsq());
}

cpx EXP(double theta)
{
  return cpx(cos(theta),sin(theta));
}

const double two_pi = 4 * acos(0);

// in:     input array
// out:    output array
// step:   {SET TO 1} (used internally)
// size:   length of the input/output {MUST BE A POWER OF 2}
// dir:    either plus or minus one (direction of the FFT)
// RESULT: out[k] = \sum_{j=0}^{size - 1} in[j] * exp(dir * 2pi * i * j * k / size)
void FFT(cpx *in, cpx *out, int step, int size, int dir)
{
  if(size < 1) return;
  if(size == 1)
  {
    out[0] = in[0];
    return;
  }
  FFT(in, out, step * 2, size / 2, dir);
  FFT(in + step, out + size / 2, step * 2, size / 2, dir);
  for(int i = 0 ; i < size / 2 ; i++)
  {
    cpx even = out[i];
    cpx odd = out[i + size / 2];
    out[i] = even + EXP(dir * two_pi * i / size) * odd;
    out[i + size / 2] = even + EXP(dir * two_pi * (i + size / 2) / size) * odd;
  }
}

// Usage:
// f[0...N-1] and g[0..N-1] are numbers
// Want to compute the convolution h, defined by
// h[n] = sum of f[k]g[n-k] (k = 0, ..., N-1).
// Here, the index is cyclic; f[-1] = f[N-1], f[-2] = f[N-2], etc.
// Let F[0...N-1] be FFT(f), and similarly, define G and H.
// The convolution theorem says H[n] = F[n]G[n] (element-wise product).
// To compute h[] in O(N log N) time, do the following:
//   1. Compute F and G (pass dir = 1 as the argument).
//   2. Get H by element-wise multiplying F and G.
//   3. Get h by taking the inverse FFT (use dir = -1 as the argument)
//      and *dividing by N*. DO NOT FORGET THIS SCALING FACTOR.

int main(void)
{
  printf("If rows come in identical pairs, then everything works.\n");
  
  cpx a[8] = {0, 1, cpx(1,3), cpx(0,5), 1, 0, 2, 0};
  cpx b[8] = {1, cpx(0,-2), cpx(0,1), 3, -1, -3, 1, -2};
  cpx A[8];
  cpx B[8];
  FFT(a, A, 1, 8, 1);
  FFT(b, B, 1, 8, 1);
  
  for(int i = 0 ; i < 8 ; i++)
  {
    printf("%7.2lf%7.2lf", A[i].a, A[i].b);
  }
  printf("\n");
  for(int i = 0 ; i < 8 ; i++)
  {
    cpx Ai(0,0);
    for(int j = 0 ; j < 8 ; j++)
    {
      Ai = Ai + a[j] * EXP(j * i * two_pi / 8);
    }
    printf("%7.2lf%7.2lf", Ai.a, Ai.b);
  }
  printf("\n");
  
  cpx AB[8];
  for(int i = 0 ; i < 8 ; i++)
    AB[i] = A[i] * B[i];
  cpx aconvb[8];
  FFT(AB, aconvb, 1, 8, -1);
  for(int i = 0 ; i < 8 ; i++)
    aconvb[i] = aconvb[i] / 8;
  for(int i = 0 ; i < 8 ; i++)
  {
    printf("%7.2lf%7.2lf", aconvb[i].a, aconvb[i].b);
  }
  printf("\n");
  for(int i = 0 ; i < 8 ; i++)
  {
    cpx aconvbi(0,0);
    for(int j = 0 ; j < 8 ; j++)
    {
      aconvbi = aconvbi + a[j] * b[(8 + i - j) % 8];
    }
    printf("%7.2lf%7.2lf", aconvbi.a, aconvbi.b);
  }
  printf("\n");
  
  return 0;
}
```
### FastExpo.cc
```cpp
/*
Uses powers of two to exponentiate numbers and matrices. Calculates
n^k in O(log(k)) time when n is a number. If A is an n x n matrix,
calculates A^k in O(n^3*log(k)) time.
*/

#include <iostream>
#include <vector>

using namespace std;

typedef double T;
typedef vector<T> VT;
typedef vector<VT> VVT;

T power(T x, int k) {
  T ret = 1;
  
  while(k) {
    if(k & 1) ret *= x;
    k >>= 1; x *= x;
  }
  return ret;
}

VVT multiply(VVT& A, VVT& B) {
  int n = A.size(), m = A[0].size(), k = B[0].size();
  VVT C(n, VT(k, 0));
  
  for(int i = 0; i < n; i++)
    for(int j = 0; j < k; j++)
      for(int l = 0; l < m; l++)
        C[i][j] += A[i][l] * B[l][j];

  return C;
}

VVT power(VVT& A, int k) {
  int n = A.size();
  VVT ret(n, VT(n)), B = A;
  for(int i = 0; i < n; i++) ret[i][i]=1;

  while(k) {
    if(k & 1) ret = multiply(ret, B);
    k >>= 1; B = multiply(B, B);
  }
  return ret;
}

int main()
{
  /* Expected Output:
     2.37^48 = 9.72569e+17
     376 264 285 220 265 
     550 376 529 285 484 
     484 265 376 264 285 
     285 220 265 156 264 
     529 285 484 265 376 */
  double n = 2.37;
  int k = 48;
  
  cout << n << "^" << k << " = " << power(n, k) << endl;
  
  double At[5][5] = {
    { 0, 0, 1, 0, 0 },
    { 1, 0, 0, 1, 0 },
    { 0, 0, 0, 0, 1 },
    { 1, 0, 0, 0, 0 },
    { 0, 1, 0, 0, 0 } };
    
  vector <vector <double> > A(5, vector <double>(5));    
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 5; j++)
      A[i][j] = At[i][j];
    
  vector <vector <double> > Ap = power(A, k);
  
  cout << endl;
  for(int i = 0; i < 5; i++) {
    for(int j = 0; j < 5; j++)
      cout << Ap[i][j] << " ";
    cout << endl;
  }
}
```
### Floyd-Warshall
```cpp
#include <iostream>
#include <queue>
#include <cmath>
#include <vector>

using namespace std;

typedef double T;
typedef vector<T> VT;
typedef vector<VT> VVT;

typedef vector<int> VI;
typedef vector<VI> VVI;

// This function runs the Floyd-Warshall algorithm for all-pairs
// shortest paths.  Also handles negative edge weights.  Returns true
// if a negative weight cycle is found.
//
// Running time: O(|V|^3)
//
//   INPUT:  w[i][j] = weight of edge from i to j
//   OUTPUT: w[i][j] = shortest path from i to j
//           prev[i][j] = node before j on the best path starting at i

bool FloydWarshall (VVT &w, VVI &prev){
  int n = w.size();
  prev = VVI (n, VI(n, -1));
  
  for (int k = 0; k < n; k++){
    for (int i = 0; i < n; i++){
      for (int j = 0; j < n; j++){
        if (w[i][j] > w[i][k] + w[k][j]){
          w[i][j] = w[i][k] + w[k][j];
          prev[i][j] = k;
        }
      }
    }
  }
 
  // check for negative weight cycles
  for(int i=0;i<n;i++)
    if (w[i][i] < 0) return false;
  return true;
}
```

### GaussJordan.cc
```cpp
// Gauss-Jordan elimination with full pivoting.
//
// Uses:
//   (1) solving systems of linear equations (AX=B)
//   (2) inverting matrices (AX=I)
//   (3) computing determinants of square matrices
//
// Running time: O(n^3)
//
// INPUT:    a[][] = an nxn matrix
//           b[][] = an nxm matrix
//
// OUTPUT:   X      = an nxm matrix (stored in b[][])
//           A^{-1} = an nxn matrix (stored in a[][])
//           returns determinant of a[][]

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

const double EPS = 1e-10;

typedef vector<int> VI;
typedef double T;
typedef vector<T> VT;
typedef vector<VT> VVT;

T GaussJordan(VVT &a, VVT &b) {
  const int n = a.size();
  const int m = b[0].size();
  VI irow(n), icol(n), ipiv(n);
  T det = 1;

  for (int i = 0; i < n; i++) {
    int pj = -1, pk = -1;
    for (int j = 0; j < n; j++) if (!ipiv[j])
      for (int k = 0; k < n; k++) if (!ipiv[k])
	if (pj == -1 || fabs(a[j][k]) > fabs(a[pj][pk])) { pj = j; pk = k; }
    if (fabs(a[pj][pk]) < EPS) { cerr << "Matrix is singular." << endl; exit(0); }
    ipiv[pk]++;
    swap(a[pj], a[pk]);
    swap(b[pj], b[pk]);
    if (pj != pk) det *= -1;
    irow[i] = pj;
    icol[i] = pk;

    T c = 1.0 / a[pk][pk];
    det *= a[pk][pk];
    a[pk][pk] = 1.0;
    for (int p = 0; p < n; p++) a[pk][p] *= c;
    for (int p = 0; p < m; p++) b[pk][p] *= c;
    for (int p = 0; p < n; p++) if (p != pk) {
      c = a[p][pk];
      a[p][pk] = 0;
      for (int q = 0; q < n; q++) a[p][q] -= a[pk][q] * c;
      for (int q = 0; q < m; q++) b[p][q] -= b[pk][q] * c;      
    }
  }

  for (int p = n-1; p >= 0; p--) if (irow[p] != icol[p]) {
    for (int k = 0; k < n; k++) swap(a[k][irow[p]], a[k][icol[p]]);
  }

  return det;
}

int main() {
  const int n = 4;
  const int m = 2;
  double A[n][n] = { {1,2,3,4},{1,0,1,0},{5,3,2,4},{6,1,4,6} };
  double B[n][m] = { {1,2},{4,3},{5,6},{8,7} };
  VVT a(n), b(n);
  for (int i = 0; i < n; i++) {
    a[i] = VT(A[i], A[i] + n);
    b[i] = VT(B[i], B[i] + m);
  }
  
  double det = GaussJordan(a, b);
  
  // expected: 60  
  cout << "Determinant: " << det << endl;

  // expected: -0.233333 0.166667 0.133333 0.0666667
  //           0.166667 0.166667 0.333333 -0.333333
  //           0.233333 0.833333 -0.133333 -0.0666667
  //           0.05 -0.75 -0.1 0.2
  cout << "Inverse: " << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      cout << a[i][j] << ' ';
    cout << endl;
  }
  
  // expected: 1.63333 1.3
  //           -0.166667 0.5
  //           2.36667 1.7
  //           -1.85 -1.35
  cout << "Solution: " << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
      cout << b[i][j] << ' ';
    cout << endl;
  }
}
```
### Geometry.cc
```cpp
// C++ routines for computational geometry.

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

double INF = 1e100;
double EPS = 1e-12;

struct PT { 
  double x, y; 
  PT() {}
  PT(double x, double y) : x(x), y(y) {}
  PT(const PT &p) : x(p.x), y(p.y)    {}
  PT operator + (const PT &p)  const { return PT(x+p.x, y+p.y); }
  PT operator - (const PT &p)  const { return PT(x-p.x, y-p.y); }
  PT operator * (double c)     const { return PT(x*c,   y*c  ); }
  PT operator / (double c)     const { return PT(x/c,   y/c  ); }
};

double dot(PT p, PT q)     { return p.x*q.x+p.y*q.y; }
double dist2(PT p, PT q)   { return dot(p-q,p-q); }
double cross(PT p, PT q)   { return p.x*q.y-p.y*q.x; }
ostream &operator<<(ostream &os, const PT &p) {
  os << "(" << p.x << "," << p.y << ")"; 
}

// rotate a point CCW or CW around the origin
PT RotateCCW90(PT p)   { return PT(-p.y,p.x); }
PT RotateCW90(PT p)    { return PT(p.y,-p.x); }
PT RotateCCW(PT p, double t) { 
  return PT(p.x*cos(t)-p.y*sin(t), p.x*sin(t)+p.y*cos(t)); 
}

// project point c onto line through a and b
// assuming a != b
PT ProjectPointLine(PT a, PT b, PT c) {
  return a + (b-a)*dot(c-a, b-a)/dot(b-a, b-a);
}

// project point c onto line segment through a and b
PT ProjectPointSegment(PT a, PT b, PT c) {
  double r = dot(b-a,b-a);
  if (fabs(r) < EPS) return a;
  r = dot(c-a, b-a)/r;
  if (r < 0) return a;
  if (r > 1) return b;
  return a + (b-a)*r;
}

// compute distance from c to segment between a and b
double DistancePointSegment(PT a, PT b, PT c) {
  return sqrt(dist2(c, ProjectPointSegment(a, b, c)));
}

// compute distance between point (x,y,z) and plane ax+by+cz=d
double DistancePointPlane(double x, double y, double z,
                          double a, double b, double c, double d)
{
  return fabs(a*x+b*y+c*z-d)/sqrt(a*a+b*b+c*c);
}

// determine if lines from a to b and c to d are parallel or collinear
bool LinesParallel(PT a, PT b, PT c, PT d) { 
  return fabs(cross(b-a, c-d)) < EPS; 
}

bool LinesCollinear(PT a, PT b, PT c, PT d) { 
  return LinesParallel(a, b, c, d)
      && fabs(cross(a-b, a-c)) < EPS
      && fabs(cross(c-d, c-a)) < EPS; 
}

// determine if line segment from a to b intersects with 
// line segment from c to d
bool SegmentsIntersect(PT a, PT b, PT c, PT d) {
  if (LinesCollinear(a, b, c, d)) {
    if (dist2(a, c) < EPS || dist2(a, d) < EPS ||
      dist2(b, c) < EPS || dist2(b, d) < EPS) return true;
    if (dot(c-a, c-b) > 0 && dot(d-a, d-b) > 0 && dot(c-b, d-b) > 0)
      return false;
    return true;
  }
  if (cross(d-a, b-a) * cross(c-a, b-a) > 0) return false;
  if (cross(a-c, d-c) * cross(b-c, d-c) > 0) return false;
  return true;
}

// compute intersection of line passing through a and b
// with line passing through c and d, assuming that unique
// intersection exists; for segment intersection, check if
// segments intersect first
PT ComputeLineIntersection(PT a, PT b, PT c, PT d) {
  b=b-a; d=c-d; c=c-a;
  assert(dot(b, b) > EPS && dot(d, d) > EPS);
  return a + b*cross(c, d)/cross(b, d);
}

// compute center of circle given three points
PT ComputeCircleCenter(PT a, PT b, PT c) {
  b=(a+b)/2;
  c=(a+c)/2;
  return ComputeLineIntersection(b, b+RotateCW90(a-b), c, c+RotateCW90(a-c));
}

// determine if point is in a possibly non-convex polygon (by William
// Randolph Franklin); returns 1 for strictly interior points, 0 for
// strictly exterior points, and 0 or 1 for the remaining points.
// Note that it is possible to convert this into an *exact* test using
// integer arithmetic by taking care of the division appropriately
// (making sure to deal with signs properly) and then by writing exact
// tests for checking point on polygon boundary
bool PointInPolygon(const vector<PT> &p, PT q) {
  bool c = 0;
  for (int i = 0; i < p.size(); i++){
    int j = (i+1)%p.size();
    if ((p[i].y <= q.y && q.y < p[j].y || 
      p[j].y <= q.y && q.y < p[i].y) &&
      q.x < p[i].x + (p[j].x - p[i].x) * (q.y - p[i].y) / (p[j].y - p[i].y))
      c = !c;
  }
  return c;
}

// determine if point is on the boundary of a polygon
bool PointOnPolygon(const vector<PT> &p, PT q) {
  for (int i = 0; i < p.size(); i++)
    if (dist2(ProjectPointSegment(p[i], p[(i+1)%p.size()], q), q) < EPS)
      return true;
    return false;
}

// compute intersection of line through points a and b with
// circle centered at c with radius r > 0
vector<PT> CircleLineIntersection(PT a, PT b, PT c, double r) {
  vector<PT> ret;
  b = b-a;
  a = a-c;
  double A = dot(b, b);
  double B = dot(a, b);
  double C = dot(a, a) - r*r;
  double D = B*B - A*C;
  if (D < -EPS) return ret;
  ret.push_back(c+a+b*(-B+sqrt(D+EPS))/A);
  if (D > EPS)
    ret.push_back(c+a+b*(-B-sqrt(D))/A);
  return ret;
}

// compute intersection of circle centered at a with radius r
// with circle centered at b with radius R
vector<PT> CircleCircleIntersection(PT a, PT b, double r, double R) {
  vector<PT> ret;
  double d = sqrt(dist2(a, b));
  if (d > r+R || d+min(r, R) < max(r, R)) return ret;
  double x = (d*d-R*R+r*r)/(2*d);
  double y = sqrt(r*r-x*x);
  PT v = (b-a)/d;
  ret.push_back(a+v*x + RotateCCW90(v)*y);
  if (y > 0)
    ret.push_back(a+v*x - RotateCCW90(v)*y);
  return ret;
}

// This code computes the area or centroid of a (possibly nonconvex)
// polygon, assuming that the coordinates are listed in a clockwise or
// counterclockwise fashion.  Note that the centroid is often known as
// the "center of gravity" or "center of mass".
double ComputeSignedArea(const vector<PT> &p) {
  double area = 0;
  for(int i = 0; i < p.size(); i++) {
    int j = (i+1) % p.size();
    area += p[i].x*p[j].y - p[j].x*p[i].y;
  }
  return area / 2.0;
}

double ComputeArea(const vector<PT> &p) {
  return fabs(ComputeSignedArea(p));
}

PT ComputeCentroid(const vector<PT> &p) {
  PT c(0,0);
  double scale = 6.0 * ComputeSignedArea(p);
  for (int i = 0; i < p.size(); i++){
    int j = (i+1) % p.size();
    c = c + (p[i]+p[j])*(p[i].x*p[j].y - p[j].x*p[i].y);
  }
  return c / scale;
}

// tests whether or not a given polygon (in CW or CCW order) is simple
bool IsSimple(const vector<PT> &p) {
  for (int i = 0; i < p.size(); i++) {
    for (int k = i+1; k < p.size(); k++) {
      int j = (i+1) % p.size();
      int l = (k+1) % p.size();
      if (i == l || j == k) continue;
      if (SegmentsIntersect(p[i], p[j], p[k], p[l])) 
        return false;
    }
  }
  return true;
}

int main() {
  
  // expected: (-5,2)
  cerr << RotateCCW90(PT(2,5)) << endl;
  
  // expected: (5,-2)
  cerr << RotateCW90(PT(2,5)) << endl;
  
  // expected: (-5,2)
  cerr << RotateCCW(PT(2,5),M_PI/2) << endl;
  
  // expected: (5,2)
  cerr << ProjectPointLine(PT(-5,-2), PT(10,4), PT(3,7)) << endl;
  
  // expected: (5,2) (7.5,3) (2.5,1)
  cerr << ProjectPointSegment(PT(-5,-2), PT(10,4), PT(3,7)) << " "
       << ProjectPointSegment(PT(7.5,3), PT(10,4), PT(3,7)) << " "
       << ProjectPointSegment(PT(-5,-2), PT(2.5,1), PT(3,7)) << endl;
  
  // expected: 6.78903
  cerr << DistancePointPlane(4,-4,3,2,-2,5,-8) << endl;
  
  // expected: 1 0 1
  cerr << LinesParallel(PT(1,1), PT(3,5), PT(2,1), PT(4,5)) << " "
       << LinesParallel(PT(1,1), PT(3,5), PT(2,0), PT(4,5)) << " "
       << LinesParallel(PT(1,1), PT(3,5), PT(5,9), PT(7,13)) << endl;
  
  // expected: 0 0 1
  cerr << LinesCollinear(PT(1,1), PT(3,5), PT(2,1), PT(4,5)) << " "
       << LinesCollinear(PT(1,1), PT(3,5), PT(2,0), PT(4,5)) << " "
       << LinesCollinear(PT(1,1), PT(3,5), PT(5,9), PT(7,13)) << endl;
  
  // expected: 1 1 1 0
  cerr << SegmentsIntersect(PT(0,0), PT(2,4), PT(3,1), PT(-1,3)) << " "
       << SegmentsIntersect(PT(0,0), PT(2,4), PT(4,3), PT(0,5)) << " "
       << SegmentsIntersect(PT(0,0), PT(2,4), PT(2,-1), PT(-2,1)) << " "
       << SegmentsIntersect(PT(0,0), PT(2,4), PT(5,5), PT(1,7)) << endl;
  
  // expected: (1,2)
  cerr << ComputeLineIntersection(PT(0,0), PT(2,4), PT(3,1), PT(-1,3)) << endl;
  
  // expected: (1,1)
  cerr << ComputeCircleCenter(PT(-3,4), PT(6,1), PT(4,5)) << endl;
  
  vector<PT> v; 
  v.push_back(PT(0,0));
  v.push_back(PT(5,0));
  v.push_back(PT(5,5));
  v.push_back(PT(0,5));
  
  // expected: 1 1 1 0 0
  cerr << PointInPolygon(v, PT(2,2)) << " "
       << PointInPolygon(v, PT(2,0)) << " "
       << PointInPolygon(v, PT(0,2)) << " "
       << PointInPolygon(v, PT(5,2)) << " "
       << PointInPolygon(v, PT(2,5)) << endl;
  
  // expected: 0 1 1 1 1
  cerr << PointOnPolygon(v, PT(2,2)) << " "
       << PointOnPolygon(v, PT(2,0)) << " "
       << PointOnPolygon(v, PT(0,2)) << " "
       << PointOnPolygon(v, PT(5,2)) << " "
       << PointOnPolygon(v, PT(2,5)) << endl;
  
  // expected: (1,6)
  //           (5,4) (4,5)
  //           blank line
  //           (4,5) (5,4)
  //           blank line
  //           (4,5) (5,4)
  vector<PT> u = CircleLineIntersection(PT(0,6), PT(2,6), PT(1,1), 5);
  for (int i = 0; i < u.size(); i++) cerr << u[i] << " "; cerr << endl;
  u = CircleLineIntersection(PT(0,9), PT(9,0), PT(1,1), 5);
  for (int i = 0; i < u.size(); i++) cerr << u[i] << " "; cerr << endl;
  u = CircleCircleIntersection(PT(1,1), PT(10,10), 5, 5);
  for (int i = 0; i < u.size(); i++) cerr << u[i] << " "; cerr << endl;
  u = CircleCircleIntersection(PT(1,1), PT(8,8), 5, 5);
  for (int i = 0; i < u.size(); i++) cerr << u[i] << " "; cerr << endl;
  u = CircleCircleIntersection(PT(1,1), PT(4.5,4.5), 10, sqrt(2.0)/2.0);
  for (int i = 0; i < u.size(); i++) cerr << u[i] << " "; cerr << endl;
  u = CircleCircleIntersection(PT(1,1), PT(4.5,4.5), 5, sqrt(2.0)/2.0);
  for (int i = 0; i < u.size(); i++) cerr << u[i] << " "; cerr << endl;
  
  // area should be 5.0
  // centroid should be (1.1666666, 1.166666)
  PT pa[] = { PT(0,0), PT(5,0), PT(1,1), PT(0,5) };
  vector<PT> p(pa, pa+4);
  PT c = ComputeCentroid(p);
  cerr << "Area: " << ComputeArea(p) << endl;
  cerr << "Centroid: " << c << endl;
  
  return 0;
}
```
### GraphCutInference.cc
```cpp
// Special-purpose {0,1} combinatorial optimization solver for
// problems of the following by a reduction to graph cuts:
//
//        minimize         sum_i  psi_i(x[i]) 
//  x[1]...x[n] in {0,1}      + sum_{i < j}  phi_{ij}(x[i], x[j])
//
// where
//      psi_i : {0, 1} --> R
//   phi_{ij} : {0, 1} x {0, 1} --> R
//
// such that
//   phi_{ij}(0,0) + phi_{ij}(1,1) <= phi_{ij}(0,1) + phi_{ij}(1,0)  (*)
//
// This can also be used to solve maximization problems where the
// direction of the inequality in (*) is reversed.
//
// INPUT: phi -- a matrix such that phi[i][j][u][v] = phi_{ij}(u, v)
//        psi -- a matrix such that psi[i][u] = psi_i(u)
//        x -- a vector where the optimal solution will be stored
//
// OUTPUT: value of the optimal solution
//
// To use this code, create a GraphCutInference object, and call the
// DoInference() method.  To perform maximization instead of minimization,
// ensure that #define MAXIMIZATION is enabled.

#include <vector>
#include <iostream>

using namespace std;

typedef vector<int> VI;
typedef vector<VI> VVI;
typedef vector<VVI> VVVI;
typedef vector<VVVI> VVVVI;

const int INF = 1000000000;

// comment out following line for minimization
#define MAXIMIZATION

struct GraphCutInference {
  int N;
  VVI cap, flow;
  VI reached;
  
  int Augment(int s, int t, int a) {
    reached[s] = 1;
    if (s == t) return a; 
    for (int k = 0; k < N; k++) {
      if (reached[k]) continue;
      if (int aa = min(a, cap[s][k] - flow[s][k])) {
	if (int b = Augment(k, t, aa)) {
	  flow[s][k] += b;
	  flow[k][s] -= b;
	  return b;
	}
      }
    }
    return 0;
  }
  
  int GetMaxFlow(int s, int t) {
    N = cap.size();
    flow = VVI(N, VI(N));
    reached = VI(N);
    
    int totflow = 0;
    while (int amt = Augment(s, t, INF)) {
      totflow += amt;
      fill(reached.begin(), reached.end(), 0);
    }
    return totflow;
  }
  
  int DoInference(const VVVVI &phi, const VVI &psi, VI &x) {
    int M = phi.size();
    cap = VVI(M+2, VI(M+2));
    VI b(M);
    int c = 0;

    for (int i = 0; i < M; i++) {
      b[i] += psi[i][1] - psi[i][0];
      c += psi[i][0];
      for (int j = 0; j < i; j++)
	b[i] += phi[i][j][1][1] - phi[i][j][0][1];
      for (int j = i+1; j < M; j++) {
	cap[i][j] = phi[i][j][0][1] + phi[i][j][1][0] - phi[i][j][0][0] - phi[i][j][1][1];
	b[i] += phi[i][j][1][0] - phi[i][j][0][0];
	c += phi[i][j][0][0];
      }
    }
    
#ifdef MAXIMIZATION
    for (int i = 0; i < M; i++) {
      for (int j = i+1; j < M; j++) 
	cap[i][j] *= -1;
      b[i] *= -1;
    }
    c *= -1;
#endif

    for (int i = 0; i < M; i++) {
      if (b[i] >= 0) {
	cap[M][i] = b[i];
      } else {
	cap[i][M+1] = -b[i];
	c += b[i];
      }
    }

    int score = GetMaxFlow(M, M+1);
    fill(reached.begin(), reached.end(), 0);
    Augment(M, M+1, INF);
    x = VI(M);
    for (int i = 0; i < M; i++) x[i] = reached[i] ? 0 : 1;
    score += c;
#ifdef MAXIMIZATION
    score *= -1;
#endif

    return score;
  }

};

int main() {

  // solver for "Cat vs. Dog" from NWERC 2008
  
  int numcases;
  cin >> numcases;
  for (int caseno = 0; caseno < numcases; caseno++) {
    int c, d, v;
    cin >> c >> d >> v;

    VVVVI phi(c+d, VVVI(c+d, VVI(2, VI(2))));
    VVI psi(c+d, VI(2));
    for (int i = 0; i < v; i++) {
      char p, q;
      int u, v;
      cin >> p >> u >> q >> v;
      u--; v--;
      if (p == 'C') {
	phi[u][c+v][0][0]++;
	phi[c+v][u][0][0]++;
      } else {
	phi[v][c+u][1][1]++;
	phi[c+u][v][1][1]++;
      }
    }
    
    GraphCutInference graph;
    VI x;
    cout << graph.DoInference(phi, psi, x) << endl;
  }

  return 0;
}
```
### IO.cc
```cpp
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    // Ouput a specific number of digits past the decimal point,
    // in this case 5    
    cout.setf(ios::fixed); cout << setprecision(5);
    cout << 100.0/7.0 << endl;
    cout.unsetf(ios::fixed);
    
    // Output the decimal point and trailing zeros
    cout.setf(ios::showpoint);
    cout << 100.0 << endl;
    cout.unsetf(ios::showpoint);
    
    // Output a '+' before positive values
    cout.setf(ios::showpos);
    cout << 100 << " " << -100 << endl;
    cout.unsetf(ios::showpos);
    
    // Output numerical values in hexadecimal
    cout << hex << 100 << " " << 1000 << " " << 10000 << dec << endl;
}
```
### KDTree.cc
```cpp
// -----------------------------------------------------------------
// A straightforward, but probably sub-optimal KD-tree implmentation
// that's probably good enough for most things (current it's a
// 2D-tree)
//
//  - constructs from n points in O(n lg^2 n) time
//  - handles nearest-neighbor query in O(lg n) if points are well
//    distributed
//  - worst case for nearest-neighbor may be linear in pathological
//    case
//
// Sonny Chan, Stanford University, April 2009
// -----------------------------------------------------------------

#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>

using namespace std;

// number type for coordinates, and its maximum value
typedef long long ntype;
const ntype sentry = numeric_limits<ntype>::max();

// point structure for 2D-tree, can be extended to 3D
struct point {
    ntype x, y;
    point(ntype xx = 0, ntype yy = 0) : x(xx), y(yy) {}
};

bool operator==(const point &a, const point &b)
{
    return a.x == b.x && a.y == b.y;
}

// sorts points on x-coordinate
bool on_x(const point &a, const point &b)
{
    return a.x < b.x;
}

// sorts points on y-coordinate
bool on_y(const point &a, const point &b)
{
    return a.y < b.y;
}

// squared distance between points
ntype pdist2(const point &a, const point &b)
{
    ntype dx = a.x-b.x, dy = a.y-b.y;
    return dx*dx + dy*dy;
}

// bounding box for a set of points
struct bbox
{
    ntype x0, x1, y0, y1;
    
    bbox() : x0(sentry), x1(-sentry), y0(sentry), y1(-sentry) {}
    
    // computes bounding box from a bunch of points
    void compute(const vector<point> &v) {
        for (int i = 0; i < v.size(); ++i) {
            x0 = min(x0, v[i].x);   x1 = max(x1, v[i].x);
            y0 = min(y0, v[i].y);   y1 = max(y1, v[i].y);
        }
    }
    
    // squared distance between a point and this bbox, 0 if inside
    ntype distance(const point &p) {
        if (p.x < x0) {
            if (p.y < y0)       return pdist2(point(x0, y0), p);
            else if (p.y > y1)  return pdist2(point(x0, y1), p);
            else                return pdist2(point(x0, p.y), p);
        }
        else if (p.x > x1) {
            if (p.y < y0)       return pdist2(point(x1, y0), p);
            else if (p.y > y1)  return pdist2(point(x1, y1), p);
            else                return pdist2(point(x1, p.y), p);
        }
        else {
            if (p.y < y0)       return pdist2(point(p.x, y0), p);
            else if (p.y > y1)  return pdist2(point(p.x, y1), p);
            else                return 0;
        }
    }
};

// stores a single node of the kd-tree, either internal or leaf
struct kdnode 
{
    bool leaf;      // true if this is a leaf node (has one point)
    point pt;       // the single point of this is a leaf
    bbox bound;     // bounding box for set of points in children
    
    kdnode *first, *second; // two children of this kd-node
    
    kdnode() : leaf(false), first(0), second(0) {}
    ~kdnode() { if (first) delete first; if (second) delete second; }
    
    // intersect a point with this node (returns squared distance)
    ntype intersect(const point &p) {
        return bound.distance(p);
    }
    
    // recursively builds a kd-tree from a given cloud of points
    void construct(vector<point> &vp)
    {
        // compute bounding box for points at this node
        bound.compute(vp);
        
        // if we're down to one point, then we're a leaf node
        if (vp.size() == 1) {
            leaf = true;
            pt = vp[0];
        }
        else {
            // split on x if the bbox is wider than high (not best heuristic...)
            if (bound.x1-bound.x0 >= bound.y1-bound.y0)
                sort(vp.begin(), vp.end(), on_x);
            // otherwise split on y-coordinate
            else
                sort(vp.begin(), vp.end(), on_y);
            
            // divide by taking half the array for each child
            // (not best performance if many duplicates in the middle)
            int half = vp.size()/2;
            vector<point> vl(vp.begin(), vp.begin()+half);
            vector<point> vr(vp.begin()+half, vp.end());
            first = new kdnode();   first->construct(vl);
            second = new kdnode();  second->construct(vr);            
        }
    }
};

// simple kd-tree class to hold the tree and handle queries
struct kdtree
{
    kdnode *root;
    
    // constructs a kd-tree from a points (copied here, as it sorts them)
    kdtree(const vector<point> &vp) {
        vector<point> v(vp.begin(), vp.end());
        root = new kdnode();
        root->construct(v);
    }
    ~kdtree() { delete root; }
    
    // recursive search method returns squared distance to nearest point
    ntype search(kdnode *node, const point &p)
    {
        if (node->leaf) {
            // commented special case tells a point not to find itself
//            if (p == node->pt) return sentry;
//            else               
                return pdist2(p, node->pt);
        }
        
        ntype bfirst = node->first->intersect(p);
        ntype bsecond = node->second->intersect(p);
        
        // choose the side with the closest bounding box to search first
        // (note that the other side is also searched if needed)
        if (bfirst < bsecond) {
            ntype best = search(node->first, p);
            if (bsecond < best)
                best = min(best, search(node->second, p));
            return best;
        }
        else {
            ntype best = search(node->second, p);
            if (bfirst < best)
                best = min(best, search(node->first, p));
            return best;
        }
    }
    
    // squared distance to the nearest 
    ntype nearest(const point &p) {
        return search(root, p);
    }
};

// --------------------------------------------------------------------------
// some basic test code here

int main()
{
    // generate some random points for a kd-tree
    vector<point> vp;
    for (int i = 0; i < 100000; ++i) {
        vp.push_back(point(rand()%100000, rand()%100000));
    }
    kdtree tree(vp);
    
    // query some points
    for (int i = 0; i < 10; ++i) {
        point q(rand()%100000, rand()%100000);
        cout << "Closest squared distance to (" << q.x << ", " << q.y << ")"
             << " is " << tree.nearest(q) << endl;
    }    

    return 0;
}

// --------------------------------------------------------------------------
```
### KMP.cc
```cpp
/*
Searches for the string w in the string s (of length k). Returns the
0-based index of the first match (k if no match is found). Algorithm
runs in O(k) time.
*/

#include <iostream>
#include <string>
#include <vector>

using namespace std;

typedef vector<int> VI;

void buildTable(string& w, VI& t)
{
  t = VI(w.length());  
  int i = 2, j = 0;
  t[0] = -1; t[1] = 0;
  
  while(i < w.length())
  {
    if(w[i-1] == w[j]) { t[i] = j+1; i++; j++; }
    else if(j > 0) j = t[j];
    else { t[i] = 0; i++; }
  }
}

int KMP(string& s, string& w)
{
  int m = 0, i = 0;
  VI t;
  
  buildTable(w, t);  
  while(m+i < s.length())
  {
    if(w[i] == s[m+i])
    {
      i++;
      if(i == w.length()) return m;
    }
    else
    {
      m += i-t[i];
      if(i > 0) i = t[i];
    }
  }  
  return s.length();
}

int main()
{
  string a = (string) "The example above illustrates the general technique for assembling "+
    "the table with a minimum of fuss. The principle is that of the overall search: "+
    "most of the work was already done in getting to the current position, so very "+
    "little needs to be done in leaving it. The only minor complication is that the "+
    "logic which is correct late in the string erroneously gives non-proper "+
    "substrings at the beginning. This necessitates some initialization code.";
  
  string b = "table";
  
  int p = KMP(a, b);
  cout << p << ": " << a.substr(p, b.length()) << " " << b << endl;
}
```
### Kruskal.cc
```cpp
/*
Uses Kruskal's Algorithm to calculate the weight of the minimum spanning
forest (union of minimum spanning trees of each connected component) of
a possibly disjoint graph, given in the form of a matrix of edge weights
(-1 if no edge exists). Returns the weight of the minimum spanning
forest (also calculates the actual edges - stored in T). Note: uses a
disjoint-set data structure with amortized (effectively) constant time per
union/find. Runs in O(E*log(E)) time.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

typedef int T;

struct edge
{
  int u, v;
  T d;
};

struct edgeCmp
{
  int operator()(const edge& a, const edge& b) { return a.d > b.d; }
};

int find(vector <int>& C, int x) { return (C[x] == x) ? x : C[x] = find(C, C[x]); }

T Kruskal(vector <vector <T> >& w)
{
  int n = w.size();
  T weight = 0;
  
  vector <int> C(n), R(n);
  for(int i=0; i<n; i++) { C[i] = i; R[i] = 0; }
  
  vector <edge> T;
  priority_queue <edge, vector <edge>, edgeCmp> E;
  
  for(int i=0; i<n; i++)
    for(int j=i+1; j<n; j++)
      if(w[i][j] >= 0)
      {
        edge e;
        e.u = i; e.v = j; e.d = w[i][j];
        E.push(e);
      }
      
  while(T.size() < n-1 && !E.empty())
  {
    edge cur = E.top(); E.pop();
    
    int uc = find(C, cur.u), vc = find(C, cur.v);
    if(uc != vc)
    {
      T.push_back(cur); weight += cur.d;
      
      if(R[uc] > R[vc]) C[vc] = uc;
      else if(R[vc] > R[uc]) C[uc] = vc;
      else { C[vc] = uc; R[uc]++; }
    }
  }
  
  return weight;
}

int main()
{
  int wa[6][6] = {
    { 0, -1, 2, -1, 7, -1 },
    { -1, 0, -1, 2, -1, -1 },
    { 2, -1, 0, -1, 8, 6 },
    { -1, 2, -1, 0, -1, -1 },
    { 7, -1, 8, -1, 0, 4 },
    { -1, -1, 6, -1, 4, 0 } };
    
  vector <vector <int> > w(6, vector <int>(6));
  
  for(int i=0; i<6; i++)
    for(int j=0; j<6; j++)
      w[i][j] = wa[i][j];
    
  cout << Kruskal(w) << endl;
  cin >> wa[0][0];
}
```
### LCA.cc
```cpp
const int max_nodes, log_max_nodes;
int num_nodes, log_num_nodes, root;

vector<int> children[max_nodes];	// children[i] contains the children of node i
int A[max_nodes][log_max_nodes+1];	// A[i][j] is the 2^j-th ancestor of node i, or -1 if that ancestor does not exist
int L[max_nodes];			// L[i] is the distance between node i and the root

// floor of the binary logarithm of n
int lb(unsigned int n)
{
    if(n==0)
	return -1;
    int p = 0;
    if (n >= 1<<16) { n >>= 16; p += 16; }
    if (n >= 1<< 8) { n >>=  8; p +=  8; }
    if (n >= 1<< 4) { n >>=  4; p +=  4; }
    if (n >= 1<< 2) { n >>=  2; p +=  2; }
    if (n >= 1<< 1) {           p +=  1; }
    return p;
}

void DFS(int i, int l)
{
    L[i] = l;
    for(int j = 0; j < children[i].size(); j++)
	DFS(children[i][j], l+1);
}

int LCA(int p, int q)
{
    // ensure node p is at least as deep as node q
    if(L[p] < L[q])
	swap(p, q);

    // "binary search" for the ancestor of node p situated on the same level as q
    for(int i = log_num_nodes; i >= 0; i--)
	if(L[p] - (1<<i) >= L[q])
	    p = A[p][i];
    
    if(p == q)
	return p;

    // "binary search" for the LCA
    for(int i = log_num_nodes; i >= 0; i--)
	if(A[p][i] != -1 && A[p][i] != A[q][i])
	{
	    p = A[p][i];
	    q = A[q][i];
	}
    
    return A[p][0];
}

int main(int argc,char* argv[])
{
    // read num_nodes, the total number of nodes
    log_num_nodes=lb(num_nodes);
    
    for(int i = 0; i < num_nodes; i++)
    {
	int p;
	// read p, the parent of node i or -1 if node i is the root

	A[i][0] = p;
	if(p != -1)
	    children[p].push_back(i);
	else
	    root = i;
    }

    // precompute A using dynamic programming
    for(int j = 1; j <= log_num_nodes; j++)
	for(int i = 0; i < num_nodes; i++)
	    if(A[i][j-1] != -1)
		A[i][j] = A[A[i][j-1]][j-1];
	    else
		A[i][j] = -1;

    // precompute L
    DFS(root, 0);

    
    return 0;
}
```
### LCS.cc
```cpp
/*
Calculates the length of the longest common subsequence of two vectors.
Backtracks to find a single subsequence or all subsequences. Runs in
O(m*n) time except for finding all longest common subsequences, which
may be slow depending on how many there are.
*/

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

typedef int T;
typedef vector<T> VT;
typedef vector<VT> VVT;

typedef vector<int> VI;
typedef vector<VI> VVI;

void backtrack(VVI& dp, VT& res, VT& A, VT& B, int i, int j)
{
  if(!i || !j) return;
  if(A[i-1] == B[j-1]) { res.push_back(A[i-1]); backtrack(dp, res, A, B, i-1, j-1); }
  else
  {
    if(dp[i][j-1] >= dp[i-1][j]) backtrack(dp, res, A, B, i, j-1);
    else backtrack(dp, res, A, B, i-1, j);
  }
}

void backtrackall(VVI& dp, set<VT>& res, VT& A, VT& B, int i, int j)
{
  if(!i || !j) { res.insert(VI()); return; }  
  if(A[i-1] == B[j-1])
  {
    set<VT> tempres;
    backtrackall(dp, tempres, A, B, i-1, j-1);
    for(set<VT>::iterator it=tempres.begin(); it!=tempres.end(); it++)
    {
      VT temp = *it;
      temp.push_back(A[i-1]);
      res.insert(temp);
    }
  }
  else
  {
    if(dp[i][j-1] >= dp[i-1][j]) backtrackall(dp, res, A, B, i, j-1);
    if(dp[i][j-1] <= dp[i-1][j]) backtrackall(dp, res, A, B, i-1, j);
  }
}

VT LCS(VT& A, VT& B)
{
  VVI dp;
  int n = A.size(), m = B.size();
  dp.resize(n+1);
  for(int i=0; i<=n; i++) dp[i].resize(m+1, 0);
  
  for(int i=1; i<=n; i++)
    for(int j=1; j<=m; j++)
    {
      if(A[i-1] == B[j-1]) dp[i][j] = dp[i-1][j-1]+1;
      else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
    }
    
  VT res;
  backtrack(dp, res, A, B, n, m);
  reverse(res.begin(), res.end());
  return res;
}

set<VT> LCSall(VT& A, VT& B)
{
  VVI dp;
  int n = A.size(), m = B.size();
  dp.resize(n+1);
  for(int i=0; i<=n; i++) dp[i].resize(m+1, 0);
  for(int i=1; i<=n; i++)
    for(int j=1; j<=m; j++)
    {
      if(A[i-1] == B[j-1]) dp[i][j] = dp[i-1][j-1]+1;
      else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
    }
  set<VT> res;
  backtrackall(dp, res, A, B, n, m);
  return res;
}

int main()
{
  int a[] = { 0, 5, 5, 2, 1, 4, 2, 3 }, b[] = { 5, 2, 4, 3, 2, 1, 2, 1, 3 };
  VI A = VI(a, a+8), B = VI(b, b+9);
  VI C = LCS(A, B);
  
  for(int i=0; i<C.size(); i++) cout << C[i] << " ";
  cout << endl << endl;
  
  set <VI> D = LCSall(A, B);
  for(set<VI>::iterator it = D.begin(); it != D.end(); it++)
  {
    for(int i=0; i<(*it).size(); i++) cout << (*it)[i] << " ";
    cout << endl;
  }
}
```
### LatLong.cc
```cpp
/*
Converts from rectangular coordinates to latitude/longitude and vice
versa. Uses degrees (not radians).
*/

#include <iostream>
#include <cmath>

using namespace std;

struct ll
{
  double r, lat, lon;
};

struct rect
{
  double x, y, z;
};

ll convert(rect& P)
{
  ll Q;
  Q.r = sqrt(P.x*P.x+P.y*P.y+P.z*P.z);
  Q.lat = 180/M_PI*asin(P.z/Q.r);
  Q.lon = 180/M_PI*acos(P.x/sqrt(P.x*P.x+P.y*P.y));
  
  return Q;
}

rect convert(ll& Q)
{
  rect P;
  P.x = Q.r*cos(Q.lon*M_PI/180)*cos(Q.lat*M_PI/180);
  P.y = Q.r*sin(Q.lon*M_PI/180)*cos(Q.lat*M_PI/180);
  P.z = Q.r*sin(Q.lat*M_PI/180);
  
  return P;
}

int main()
{
  rect A;
  ll B;
  
  A.x = -1.0; A.y = 2.0; A.z = -3.0;
  
  B = convert(A);
  cout << B.r << " " << B.lat << " " << B.lon << endl;
  
  A = convert(B);
  cout << A.x << " " << A.y << " " << A.z << endl;
}
```
### LongestIncreasingSubstring
```cpp
// Given a list of numbers of length n, this routine extracts a 
// longest increasing subsequence.
//
// Running time: O(n log n)
//
//   INPUT: a vector of integers
//   OUTPUT: a vector containing the longest increasing subsequence

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef vector<int> VI;
typedef pair<int,int> PII;
typedef vector<PII> VPII;

#define STRICTLY_INCREASNG

VI LongestIncreasingSubsequence(VI v) {
  VPII best;
  VI dad(v.size(), -1);
  
  for (int i = 0; i < v.size(); i++) {
#ifdef STRICTLY_INCREASNG
    PII item = make_pair(v[i], 0);
    VPII::iterator it = lower_bound(best.begin(), best.end(), item);
    item.second = i;
#else
    PII item = make_pair(v[i], i);
    VPII::iterator it = upper_bound(best.begin(), best.end(), item);
#endif
    if (it == best.end()) {
      dad[i] = (best.size() == 0 ? -1 : best.back().second);
      best.push_back(item);
    } else {
      dad[i] = dad[it->second];
      *it = item;
    }
  }
  
  VI ret;
  for (int i = best.back().second; i >= 0; i = dad[i])
    ret.push_back(v[i]);
  reverse(ret.begin(), ret.end());
  return ret;
}
```
### MaxBipartiteMatching
```cpp
// This code performs maximum bipartite matching.
//
// Running time: O(|E| |V|) -- often much faster in practice
//
//   INPUT: w[i][j] = edge between row node i and column node j
//   OUTPUT: mr[i] = assignment for row node i, -1 if unassigned
//           mc[j] = assignment for column node j, -1 if unassigned
//           function returns number of matches made

#include <vector>

using namespace std;

typedef vector<int> VI;
typedef vector<VI> VVI;

bool FindMatch(int i, const VVI &w, VI &mr, VI &mc, VI &seen) {
  for (int j = 0; j < w[i].size(); j++) {
    if (w[i][j] && !seen[j]) {
      seen[j] = true;
      if (mc[j] < 0 || FindMatch(mc[j], w, mr, mc, seen)) {
        mr[i] = j;
        mc[j] = i;
        return true;
      }
    }
  }
  return false;
}

int BipartiteMatching(const VVI &w, VI &mr, VI &mc) {
  mr = VI(w.size(), -1);
  mc = VI(w[0].size(), -1);
  
  int ct = 0;
  for (int i = 0; i < w.size(); i++) {
    VI seen(w[0].size());
    if (FindMatch(i, w, mr, mc, seen)) ct++;
  }
  return ct;
}
```
### MaxFlow.cc
```cpp
// Adjacency matrix implementation of Dinic's blocking flow algorithm.
//
// Running time:
//     O(|V|^4)
//
// INPUT: 
//     - graph, constructed using AddEdge()
//     - source
//     - sink
//
// OUTPUT:
//     - maximum flow value
//     - To obtain the actual flow, look at positive values only.

#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

typedef vector<int> VI;
typedef vector<VI> VVI;

const int INF = 1000000000;

struct MaxFlow {
  int N;
  VVI cap, flow;
  VI dad, Q;

  MaxFlow(int N) :
    N(N), cap(N, VI(N)), flow(N, VI(N)), dad(N), Q(N) {}

  void AddEdge(int from, int to, int cap) {
    this->cap[from][to] += cap;
  }

  int BlockingFlow(int s, int t) {
    fill(dad.begin(), dad.end(), -1);
    dad[s] = -2;

    int head = 0, tail = 0;
    Q[tail++] = s;
    while (head < tail) {
      int x = Q[head++];
      for (int i = 0; i < N; i++) {
        if (dad[i] == -1 && cap[x][i] - flow[x][i] > 0) {
          dad[i] = x;
          Q[tail++] = i;
        }
      }
    }

    if (dad[t] == -1) return 0;

    int totflow = 0;
    for (int i = 0; i < N; i++) {
      if (dad[i] == -1) continue;
      int amt = cap[i][t] - flow[i][t];
      for (int j = i; amt && j != s; j = dad[j])
        amt = min(amt, cap[dad[j]][j] - flow[dad[j]][j]);
      if (amt == 0) continue;
      flow[i][t] += amt;
      flow[t][i] -= amt;
      for (int j = i; j != s; j = dad[j]) {
        flow[dad[j]][j] += amt;
        flow[j][dad[j]] -= amt;
      }
      totflow += amt;
    }

    return totflow;
  }

  int GetMaxFlow(int source, int sink) {
    int totflow = 0;
    while (int flow = BlockingFlow(source, sink))
      totflow += flow;
    return totflow;
  }
};

int main() {

  MaxFlow mf(5);
  mf.AddEdge(0, 1, 3);
  mf.AddEdge(0, 2, 4);
  mf.AddEdge(0, 3, 5);
  mf.AddEdge(0, 4, 5);
  mf.AddEdge(1, 2, 2);
  mf.AddEdge(2, 3, 4);
  mf.AddEdge(2, 4, 1);
  mf.AddEdge(3, 4, 10);
    
  // should print out "15"
  cout << mf.GetMaxFlow(0, 4) << endl;
}

// BEGIN CUT
// The following code solves SPOJ problem #203: Potholers (POTHOLE)

#ifdef COMMENT
int main() {
  int t;
  cin >> t;
  for (int i = 0; i < t; i++) {
    int n;
    cin >> n;
    MaxFlow mf(n);
    for (int j = 0; j < n-1; j++) {
      int m;
      cin >> m;
      for (int k = 0; k < m; k++) {
        int p;
        cin >> p;
	p--;
        int cap = (j == 0 || p == n-1) ? 1 : INF;
	mf.AddEdge(j, p, cap);
      }
    }

    cout << mf.GetMaxFlow(0, n-1) << endl;
  }
  return 0;
}
#endif

// END CUT
```
### MillerRabin.cc
```cpp
// Randomized Primality Test (Miller-Rabin):
//   Error rate: 2^(-TRIAL)
//   Almost constant time. srand is needed

#include <stdlib.h>
#define EPS 1e-7

typedef long long LL;

LL ModularMultiplication(LL a, LL b, LL m)
{
	LL ret=0, c=a;
	while(b)
	{
		if(b&1) ret=(ret+c)%m;
		b>>=1; c=(c+c)%m;
	}
	return ret;
}
LL ModularExponentiation(LL a, LL n, LL m)
{
	LL ret=1, c=a;
	while(n)
	{
		if(n&1) ret=ModularMultiplication(ret, c, m);
		n>>=1; c=ModularMultiplication(c, c, m);
	}
	return ret;
}
bool Witness(LL a, LL n)
{
	LL u=n-1;
  int t=0;
	while(!(u&1)){u>>=1; t++;}
	LL x0=ModularExponentiation(a, u, n), x1;
	for(int i=1;i<=t;i++)
	{
		x1=ModularMultiplication(x0, x0, n);
		if(x1==1 && x0!=1 && x0!=n-1) return true;
		x0=x1;
	}
	if(x0!=1) return true;
	return false;
}
LL Random(LL n)
{
  LL ret=rand(); ret*=32768;
	ret+=rand(); ret*=32768;
	ret+=rand(); ret*=32768;
	ret+=rand();
  return ret%n;
}
bool IsPrimeFast(LL n, int TRIAL)
{
  while(TRIAL--)
  {
    LL a=Random(n-2)+1;
    if(Witness(a, n)) return false;
  }
  return true;
}
```
### MinCostMatching.cc
```cpp
//////////////////////////////////////////////////////////////////////
// Min cost bipartite matching via shortest augmenting paths
//
// This is an O(n^3) implementation of a shortest augmenting path
// algorithm for finding min cost perfect matchings in dense
// graphs.  In practice, it solves 1000x1000 problems in around 1
// second.
//
//   cost[i][j] = cost for pairing left node i with right node j
//   Lmate[i] = index of right node that left node i pairs with
//   Rmate[j] = index of left node that right node j pairs with
//
// The values in cost[i][j] may be positive or negative.  To perform
// maximization, simply negate the cost[][] matrix.
//////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace std;

typedef vector<double> VD;
typedef vector<VD> VVD;
typedef vector<int> VI;

double MinCostMatching(const VVD &cost, VI &Lmate, VI &Rmate) {
  int n = int(cost.size());

  // construct dual feasible solution
  VD u(n);
  VD v(n);
  for (int i = 0; i < n; i++) {
    u[i] = cost[i][0];
    for (int j = 1; j < n; j++) u[i] = min(u[i], cost[i][j]);
  }
  for (int j = 0; j < n; j++) {
    v[j] = cost[0][j] - u[0];
    for (int i = 1; i < n; i++) v[j] = min(v[j], cost[i][j] - u[i]);
  }
  
  // construct primal solution satisfying complementary slackness
  Lmate = VI(n, -1);
  Rmate = VI(n, -1);
  int mated = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (Rmate[j] != -1) continue;
      if (fabs(cost[i][j] - u[i] - v[j]) < 1e-10) {
	Lmate[i] = j;
	Rmate[j] = i;
	mated++;
	break;
      }
    }
  }
  
  VD dist(n);
  VI dad(n);
  VI seen(n);
  
  // repeat until primal solution is feasible
  while (mated < n) {
    
    // find an unmatched left node
    int s = 0;
    while (Lmate[s] != -1) s++;
    
    // initialize Dijkstra
    fill(dad.begin(), dad.end(), -1);
    fill(seen.begin(), seen.end(), 0);
    for (int k = 0; k < n; k++) 
      dist[k] = cost[s][k] - u[s] - v[k];
    
    int j = 0;
    while (true) {
      
      // find closest
      j = -1;
      for (int k = 0; k < n; k++) {
	if (seen[k]) continue;
	if (j == -1 || dist[k] < dist[j]) j = k;
      }
      seen[j] = 1;
      
      // termination condition
      if (Rmate[j] == -1) break;
      
      // relax neighbors
      const int i = Rmate[j];
      for (int k = 0; k < n; k++) {
	if (seen[k]) continue;
	const double new_dist = dist[j] + cost[i][k] - u[i] - v[k];
	if (dist[k] > new_dist) {
	  dist[k] = new_dist;
	  dad[k] = j;
	}
      }
    }
    
    // update dual variables
    for (int k = 0; k < n; k++) {
      if (k == j || !seen[k]) continue;
      const int i = Rmate[k];
      v[k] += dist[k] - dist[j];
      u[i] -= dist[k] - dist[j];
    }
    u[s] += dist[j];
    
    // augment along path
    while (dad[j] >= 0) {
      const int d = dad[j];
      Rmate[j] = Rmate[d];
      Lmate[Rmate[j]] = j;
      j = d;
    }
    Rmate[j] = s;
    Lmate[s] = j;
    
    mated++;
  }
  
  double value = 0;
  for (int i = 0; i < n; i++)
    value += cost[i][Lmate[i]];
  
  return value;
}
```
### MinCostMaxFlow.cc
```cpp
// Implementation of min cost max flow algorithm using adjacency
// matrix (Edmonds and Karp 1972).  This implementation keeps track of
// forward and reverse edges separately (so you can set cap[i][j] !=
// cap[j][i]).  For a regular max flow, set all edge costs to 0.
//
// Running time, O(|V|^2) cost per augmentation
//     max flow:           O(|V|^3) augmentations
//     min cost max flow:  O(|V|^4 * MAX_EDGE_COST) augmentations
//     
// INPUT: 
//     - graph, constructed using AddEdge()
//     - source
//     - sink
//
// OUTPUT:
//     - (maximum flow value, minimum cost value)
//     - To obtain the actual flow, look at positive values only.

#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

typedef vector<int> VI;
typedef vector<VI> VVI;
typedef long long L;
typedef vector<L> VL;
typedef vector<VL> VVL;
typedef pair<int, int> PII;
typedef vector<PII> VPII;

const L INF = numeric_limits<L>::max() / 4;

struct MinCostMaxFlow {
  int N;
  VVL cap, flow, cost;
  VI found;
  VL dist, pi, width;
  VPII dad;

  MinCostMaxFlow(int N) : 
    N(N), cap(N, VL(N)), flow(N, VL(N)), cost(N, VL(N)), 
    found(N), dist(N), pi(N), width(N), dad(N) {}
  
  void AddEdge(int from, int to, L cap, L cost) {
    this->cap[from][to] = cap;
    this->cost[from][to] = cost;
  }
  
  void Relax(int s, int k, L cap, L cost, int dir) {
    L val = dist[s] + pi[s] - pi[k] + cost;
    if (cap && val < dist[k]) {
      dist[k] = val;
      dad[k] = make_pair(s, dir);
      width[k] = min(cap, width[s]);
    }
  }

  L Dijkstra(int s, int t) {
    fill(found.begin(), found.end(), false);
    fill(dist.begin(), dist.end(), INF);
    fill(width.begin(), width.end(), 0);
    dist[s] = 0;
    width[s] = INF;
    
    while (s != -1) {
      int best = -1;
      found[s] = true;
      for (int k = 0; k < N; k++) {
        if (found[k]) continue;
        Relax(s, k, cap[s][k] - flow[s][k], cost[s][k], 1);
        Relax(s, k, flow[k][s], -cost[k][s], -1);
        if (best == -1 || dist[k] < dist[best]) best = k;
      }
      s = best;
    }

    for (int k = 0; k < N; k++)
      pi[k] = min(pi[k] + dist[k], INF);
    return width[t];
  }

  pair<L, L> GetMaxFlow(int s, int t) {
    L totflow = 0, totcost = 0;
    while (L amt = Dijkstra(s, t)) {
      totflow += amt;
      for (int x = t; x != s; x = dad[x].first) {
        if (dad[x].second == 1) {
          flow[dad[x].first][x] += amt;
          totcost += amt * cost[dad[x].first][x];
        } else {
          flow[x][dad[x].first] -= amt;
          totcost -= amt * cost[x][dad[x].first];
        }
      }
    }
    return make_pair(totflow, totcost);
  }
};

// BEGIN CUT
// The following code solves UVA problem #10594: Data Flow

int main() {
  int N, M;

  while (scanf("%d%d", &N, &M) == 2) {
    VVL v(M, VL(3));
    for (int i = 0; i < M; i++)
      scanf("%Ld%Ld%Ld", &v[i][0], &v[i][1], &v[i][2]);
    L D, K;
    scanf("%Ld%Ld", &D, &K);

    MinCostMaxFlow mcmf(N+1);
    for (int i = 0; i < M; i++) {
      mcmf.AddEdge(int(v[i][0]), int(v[i][1]), K, v[i][2]);
      mcmf.AddEdge(int(v[i][1]), int(v[i][0]), K, v[i][2]);
    }
    mcmf.AddEdge(0, 1, D, 0);
    
    pair<L, L> res = mcmf.GetMaxFlow(0, N);

    if (res.first == D) {
      printf("%Ld\n", res.second);
    } else {
      printf("Impossible.\n");
    }
  }
  
  return 0;
}

// END CUT
```
### MinCut.cc
```cpp
// Adjacency matrix implementation of Stoer-Wagner min cut algorithm.
//
// Running time:
//     O(|V|^3)
//
// INPUT: 
//     - graph, constructed using AddEdge()
//
// OUTPUT:
//     - (min cut value, nodes in half of min cut)

#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

typedef vector<int> VI;
typedef vector<VI> VVI;

const int INF = 1000000000;

pair<int, VI> GetMinCut(VVI &weights) {
  int N = weights.size();
  VI used(N), cut, best_cut;
  int best_weight = -1;
  
  for (int phase = N-1; phase >= 0; phase--) {
    VI w = weights[0];
    VI added = used;
    int prev, last = 0;
    for (int i = 0; i < phase; i++) {
      prev = last;
      last = -1;
      for (int j = 1; j < N; j++)
	if (!added[j] && (last == -1 || w[j] > w[last])) last = j;
      if (i == phase-1) {
	for (int j = 0; j < N; j++) weights[prev][j] += weights[last][j];
	for (int j = 0; j < N; j++) weights[j][prev] = weights[prev][j];
	used[last] = true;
	cut.push_back(last);
	if (best_weight == -1 || w[last] < best_weight) {
	  best_cut = cut;
	  best_weight = w[last];
	}
      } else {
	for (int j = 0; j < N; j++)
	  w[j] += weights[last][j];
	added[last] = true;
      }
    }
  }
  return make_pair(best_weight, best_cut);
}

// BEGIN CUT
// The following code solves UVA problem #10989: Bomb, Divide and Conquer
int main() {
  int N;
  cin >> N;
  for (int i = 0; i < N; i++) {
    int n, m;
    cin >> n >> m;
    VVI weights(n, VI(n));
    for (int j = 0; j < m; j++) {
      int a, b, c;
      cin >> a >> b >> c;
      weights[a-1][b-1] = weights[b-1][a-1] = c;
    }
    pair<int, VI> res = GetMinCut(weights);
    cout << "Case #" << i+1 << ": " << res.first << endl;
  }
}
// END CUT
```
### Minimax.cc
```
template<typename T>
void child_states(const state& s, const bool maxing, T func) {
    for(/* each child state of s, for player `maxing`'s turn */)
        if (func(child))
            return;
}
// with alpha/beta pruning
int minimax(const state& s, int alpha, int beta, const bool maxing) {
    int v;
    if (maxing) {
        v = INT_MIN;
        child_states(s, maxing, [&](const state& child) {
            v = max(v, minimax(child, alpha, beta, false));
            alpha = max(alpha, v);
            return alpha >= beta;
        });
        if (v == INT_MIN)  // no child states
            return /* final score */;
    } else {
        v = INT_MAX;
        child_states(s, maxing, [&](const state& child) {
            v = min(v, minimax(b, alpha, beta, true));
            beta = min(beta, v);
            return alpha >= beta;
        });
        if (v == INT_MAX)  // no child states
            return /* final score */;
    }
    return v;
}
```
### Prim.cc
```cpp
// This function runs Prim's algorithm for constructing minimum
// weight spanning trees.
//
// Running time: O(|V|^2)
//
//   INPUT:   w[i][j] = cost of edge from i to j
//
//            NOTE: Make sure that w[i][j] is nonnegative and
//            symmetric.  Missing edges should be given -1
//            weight.
//            
//   OUTPUT:  edges = list of pair<int,int> in minimum spanning tree
//            return total weight of tree

#include <iostream>
#include <queue>
#include <cmath>
#include <vector>

using namespace std;

typedef double T;
typedef vector<T> VT;
typedef vector<VT> VVT;

typedef vector<int> VI;
typedef vector<VI> VVI;
typedef pair<int,int> PII;
typedef vector<PII> VPII;

T Prim (const VVT &w, VPII &edges){
  int n = w.size();
  VI found (n);
  VI prev (n, -1);
  VT dist (n, 1000000000);
  int here = 0;
  dist[here] = 0;
  
  while (here != -1){
    found[here] = true;
    int best = -1;
    for (int k = 0; k < n; k++) if (!found[k]){
      if (w[here][k] != -1 && dist[k] > w[here][k]){
        dist[k] = w[here][k];
        prev[k] = here;
      }
      if (best == -1 || dist[k] < dist[best]) best = k;
    }
    here = best;    
  }
  
  T tot_weight = 0;
  for (int i = 0; i < n; i++) if (prev[i] != -1){
    edges.push_back (make_pair (prev[i], i));
    tot_weight += w[prev[i]][i];
  }
  return tot_weight;  
}

int main(){
  int ww[5][5] = {
    {0, 400, 400, 300, 600},
    {400, 0, 3, -1, 7},
    {400, 3, 0, 2, 0},
    {300, -1, 2, 0, 5},
    {600, 7, 0, 5, 0}
  };
  VVT w(5, VT(5));
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      w[i][j] = ww[i][j];
    
  // expected: 305
  //           2 1
  //           3 2
  //           0 3
  //           2 4
  
  VPII edges;
  cout << Prim (w, edges) << endl;
  for (int i = 0; i < edges.size(); i++)
    cout << edges[i].first << " " << edges[i].second << endl;
}
```
### Primes.cc
```cpp
// O(sqrt(x)) Exhaustive Primality Test
#include <cmath>
#define EPS 1e-7
typedef long long LL;
bool IsPrimeSlow (LL x)
{
  if(x<=1) return false;
  if(x<=3) return true;
  if (!(x%2) || !(x%3)) return false;
  LL s=(LL)(sqrt((double)(x))+EPS);
  for(LL i=5;i<=s;i+=6)
  {
    if (!(x%i) || !(x%(i+2))) return false;
  }
  return true;
}
// Primes less than 1000:
//      2     3     5     7    11    13    17    19    23    29    31    37
//     41    43    47    53    59    61    67    71    73    79    83    89
//     97   101   103   107   109   113   127   131   137   139   149   151
//    157   163   167   173   179   181   191   193   197   199   211   223
//    227   229   233   239   241   251   257   263   269   271   277   281
//    283   293   307   311   313   317   331   337   347   349   353   359
//    367   373   379   383   389   397   401   409   419   421   431   433
//    439   443   449   457   461   463   467   479   487   491   499   503
//    509   521   523   541   547   557   563   569   571   577   587   593
//    599   601   607   613   617   619   631   641   643   647   653   659
//    661   673   677   683   691   701   709   719   727   733   739   743
//    751   757   761   769   773   787   797   809   811   821   823   827
//    829   839   853   857   859   863   877   881   883   887   907   911
//    919   929   937   941   947   953   967   971   977   983   991   997

// Other primes:
//    The largest prime smaller than 10 is 7.
//    The largest prime smaller than 100 is 97.
//    The largest prime smaller than 1000 is 997.
//    The largest prime smaller than 10000 is 9973.
//    The largest prime smaller than 100000 is 99991.
//    The largest prime smaller than 1000000 is 999983.
//    The largest prime smaller than 10000000 is 9999991.
//    The largest prime smaller than 100000000 is 99999989.
//    The largest prime smaller than 1000000000 is 999999937.
//    The largest prime smaller than 10000000000 is 9999999967.
//    The largest prime smaller than 100000000000 is 99999999977.
//    The largest prime smaller than 1000000000000 is 999999999989.
//    The largest prime smaller than 10000000000000 is 9999999999971.
//    The largest prime smaller than 100000000000000 is 99999999999973.
//    The largest prime smaller than 1000000000000000 is 999999999999989.
//    The largest prime smaller than 10000000000000000 is 9999999999999937.
//    The largest prime smaller than 100000000000000000 is 99999999999999997.
//    The largest prime smaller than 1000000000000000000 is 999999999999999989.
```
### PushRelabel.cc
```cpp
// Adjacency list implementation of FIFO push relabel maximum flow
// with the gap relabeling heuristic.  This implementation is
// significantly faster than straight Ford-Fulkerson.  It solves
// random problems with 10000 vertices and 1000000 edges in a few
// seconds, though it is possible to construct test cases that
// achieve the worst-case.
//
// Running time:
//     O(|V|^3)
//
// INPUT: 
//     - graph, constructed using AddEdge()
//     - source
//     - sink
//
// OUTPUT:
//     - maximum flow value
//     - To obtain the actual flow values, look at all edges with
//       capacity > 0 (zero capacity edges are residual edges).

#include <cmath>
#include <vector>
#include <iostream>
#include <queue>

using namespace std;

typedef long long LL;

struct Edge {
  int from, to, cap, flow, index;
  Edge(int from, int to, int cap, int flow, int index) :
    from(from), to(to), cap(cap), flow(flow), index(index) {}
};

struct PushRelabel {
  int N;
  vector<vector<Edge> > G;
  vector<LL> excess;
  vector<int> dist, active, count;
  queue<int> Q;

  PushRelabel(int N) : N(N), G(N), excess(N), dist(N), active(N), count(2*N) {}

  void AddEdge(int from, int to, int cap) {
    G[from].push_back(Edge(from, to, cap, 0, G[to].size()));
    if (from == to) G[from].back().index++;
    G[to].push_back(Edge(to, from, 0, 0, G[from].size() - 1));
  }

  void Enqueue(int v) { 
    if (!active[v] && excess[v] > 0) { active[v] = true; Q.push(v); } 
  }

  void Push(Edge &e) {
    int amt = int(min(excess[e.from], LL(e.cap - e.flow)));
    if (dist[e.from] <= dist[e.to] || amt == 0) return;
    e.flow += amt;
    G[e.to][e.index].flow -= amt;
    excess[e.to] += amt;    
    excess[e.from] -= amt;
    Enqueue(e.to);
  }
  
  void Gap(int k) {
    for (int v = 0; v < N; v++) {
      if (dist[v] < k) continue;
      count[dist[v]]--;
      dist[v] = max(dist[v], N+1);
      count[dist[v]]++;
      Enqueue(v);
    }
  }

  void Relabel(int v) {
    count[dist[v]]--;
    dist[v] = 2*N;
    for (int i = 0; i < G[v].size(); i++) 
      if (G[v][i].cap - G[v][i].flow > 0)
	dist[v] = min(dist[v], dist[G[v][i].to] + 1);
    count[dist[v]]++;
    Enqueue(v);
  }

  void Discharge(int v) {
    for (int i = 0; excess[v] > 0 && i < G[v].size(); i++) Push(G[v][i]);
    if (excess[v] > 0) {
      if (count[dist[v]] == 1) 
	Gap(dist[v]); 
      else
	Relabel(v);
    }
  }

  LL GetMaxFlow(int s, int t) {
    count[0] = N-1;
    count[N] = 1;
    dist[s] = N;
    active[s] = active[t] = true;
    for (int i = 0; i < G[s].size(); i++) {
      excess[s] += G[s][i].cap;
      Push(G[s][i]);
    }
    
    while (!Q.empty()) {
      int v = Q.front();
      Q.pop();
      active[v] = false;
      Discharge(v);
    }
    
    LL totflow = 0;
    for (int i = 0; i < G[s].size(); i++) totflow += G[s][i].flow;
    return totflow;
  }
};

// BEGIN CUT
// The following code solves SPOJ problem #4110: Fast Maximum Flow (FASTFLOW)

int main() {
  int n, m;
  scanf("%d%d", &n, &m);

  PushRelabel pr(n);
  for (int i = 0; i < m; i++) { 
   int a, b, c;
    scanf("%d%d%d", &a, &b, &c);
    if (a == b) continue;
    pr.AddEdge(a-1, b-1, c);
    pr.AddEdge(b-1, a-1, c);
  }
  printf("%Ld\n", pr.GetMaxFlow(0, n-1));
  return 0;
}

// END CUT
```
### RandomSTL.cc
```cpp
// Example for using stringstreams and next_permutation

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main(void){
  vector<int> v;
  
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  
  // Expected output: 1 2 3 4
  //                  1 2 4 3
  //                  ...
  //                  4 3 2 1  
  do {
    ostringstream oss;
    oss << v[0] << " " << v[1] << " " << v[2] << " " << v[3];
    
    // for input from a string s,
    //   istringstream iss(s);
    //   iss >> variable;
    
    cout << oss.str() << endl;
  } while (next_permutation (v.begin(), v.end()));
  
  v.clear();
  
  v.push_back(1);
  v.push_back(2);
  v.push_back(1);
  v.push_back(3);
  
  // To use unique, first sort numbers.  Then call
  // unique to place all the unique elements at the beginning
  // of the vector, and then use erase to remove the duplicate
  // elements.
  
  sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
  
  // Expected output: 1 2 3
  for (size_t i = 0; i < v.size(); i++)
    cout << v[i] << " ";
  cout << endl; 
}
```
### ReducedRowEchelonForm.cc
```cpp
// Reduced row echelon form via Gauss-Jordan elimination 
// with partial pivoting.  This can be used for computing
// the rank of a matrix.
//
// Running time: O(n^3)
//
// INPUT:    a[][] = an nxm matrix
//
// OUTPUT:   rref[][] = an nxm matrix (stored in a[][])
//           returns rank of a[][]

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

const double EPSILON = 1e-10;

typedef double T;
typedef vector<T> VT;
typedef vector<VT> VVT;

int rref(VVT &a) {
  int n = a.size();
  int m = a[0].size();
  int r = 0;
  for (int c = 0; c < m && r < n; c++) {
    int j = r;
    for (int i = r + 1; i < n; i++)
      if (fabs(a[i][c]) > fabs(a[j][c])) j = i;
    if (fabs(a[j][c]) < EPSILON) continue;
    swap(a[j], a[r]);

    T s = 1.0 / a[r][c];
    for (int j = 0; j < m; j++) a[r][j] *= s;
    for (int i = 0; i < n; i++) if (i != r) {
      T t = a[i][c];
      for (int j = 0; j < m; j++) a[i][j] -= t * a[r][j];
    }
    r++;
  }
  return r;
}

int main() {
  const int n = 5, m = 4;
  double A[n][m] = {
    {16,  2,  3, 13},
    { 5, 11, 10,  8},
    { 9,  7,  6, 12},
    { 4, 14, 15,  1},
    {13, 21, 21, 13}};
  VVT a(n);
  for (int i = 0; i < n; i++)
    a[i] = VT(A[i], A[i] + m);

  int rank = rref(a);

  // expected: 3
  cout << "Rank: " << rank << endl;

  // expected: 1 0 0 1 
  //           0 1 0 3 
  //           0 0 1 -3 
  //           0 0 0 3.10862e-15
  //           0 0 0 2.22045e-15
  cout << "rref: " << endl;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++)
      cout << a[i][j] << ' ';
    cout << endl;
  }
}
```
### SCC.cc
```cpp
#include<memory.h>
struct edge{int e, nxt;};
int V, E;
edge e[MAXE], er[MAXE];
int sp[MAXV], spr[MAXV];
int group_cnt, group_num[MAXV];
bool v[MAXV];
int stk[MAXV];
void fill_forward(int x)
{
  int i;
  v[x]=true;
  for(i=sp[x];i;i=e[i].nxt) if(!v[e[i].e]) fill_forward(e[i].e);
  stk[++stk[0]]=x;
}
void fill_backward(int x)
{
  int i;
  v[x]=false;
  group_num[x]=group_cnt;
  for(i=spr[x];i;i=er[i].nxt) if(v[er[i].e]) fill_backward(er[i].e);
}
void add_edge(int v1, int v2) //add edge v1->v2
{
  e [++E].e=v2; e [E].nxt=sp [v1]; sp [v1]=E;
  er[  E].e=v1; er[E].nxt=spr[v2]; spr[v2]=E;
}
void SCC()
{
  int i;
  stk[0]=0;
  memset(v, false, sizeof(v));
  for(i=1;i<=V;i++) if(!v[i]) fill_forward(i);
  group_cnt=0;
  for(i=stk[0];i>=1;i--) if(v[stk[i]]){group_cnt++; fill_backward(stk[i]);}
}
```
### Simplex.cc
```cpp
// Two-phase simplex algorithm for solving linear programs of the form
//
//     maximize     c^T x
//     subject to   Ax <= b
//                  x >= 0
//
// INPUT: A -- an m x n matrix
//        b -- an m-dimensional vector
//        c -- an n-dimensional vector
//        x -- a vector where the optimal solution will be stored
//
// OUTPUT: value of the optimal solution (infinity if unbounded
//         above, nan if infeasible)
//
// To use this code, create an LPSolver object with A, b, and c as
// arguments.  Then, call Solve(x).

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

typedef long double DOUBLE;
typedef vector<DOUBLE> VD;
typedef vector<VD> VVD;
typedef vector<int> VI;

const DOUBLE EPS = 1e-9;

struct LPSolver {
  int m, n;
  VI B, N;
  VVD D;

  LPSolver(const VVD &A, const VD &b, const VD &c) :
    m(b.size()), n(c.size()), N(n + 1), B(m), D(m + 2, VD(n + 2)) {
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) D[i][j] = A[i][j];
    for (int i = 0; i < m; i++) { B[i] = n + i; D[i][n] = -1; D[i][n + 1] = b[i]; }
    for (int j = 0; j < n; j++) { N[j] = j; D[m][j] = -c[j]; }
    N[n] = -1; D[m + 1][n] = 1;
  }

  void Pivot(int r, int s) {
    double inv = 1.0 / D[r][s];
    for (int i = 0; i < m + 2; i++) if (i != r)
      for (int j = 0; j < n + 2; j++) if (j != s)
        D[i][j] -= D[r][j] * D[i][s] * inv;
    for (int j = 0; j < n + 2; j++) if (j != s) D[r][j] *= inv;
    for (int i = 0; i < m + 2; i++) if (i != r) D[i][s] *= -inv;
    D[r][s] = inv;
    swap(B[r], N[s]);
  }

  bool Simplex(int phase) {
    int x = phase == 1 ? m + 1 : m;
    while (true) {
      int s = -1;
      for (int j = 0; j <= n; j++) {
        if (phase == 2 && N[j] == -1) continue;
        if (s == -1 || D[x][j] < D[x][s] || D[x][j] == D[x][s] && N[j] < N[s]) s = j;
      }
      if (D[x][s] > -EPS) return true;
      int r = -1;
      for (int i = 0; i < m; i++) {
        if (D[i][s] < EPS) continue;
        if (r == -1 || D[i][n + 1] / D[i][s] < D[r][n + 1] / D[r][s] ||
          (D[i][n + 1] / D[i][s]) == (D[r][n + 1] / D[r][s]) && B[i] < B[r]) r = i;
      }
      if (r == -1) return false;
      Pivot(r, s);
    }
  }

  DOUBLE Solve(VD &x) {
    int r = 0;
    for (int i = 1; i < m; i++) if (D[i][n + 1] < D[r][n + 1]) r = i;
    if (D[r][n + 1] < -EPS) {
      Pivot(r, n);
      if (!Simplex(1) || D[m + 1][n + 1] < -EPS) return -numeric_limits<DOUBLE>::infinity();
      for (int i = 0; i < m; i++) if (B[i] == -1) {
        int s = -1;
        for (int j = 0; j <= n; j++)
          if (s == -1 || D[i][j] < D[i][s] || D[i][j] == D[i][s] && N[j] < N[s]) s = j;
        Pivot(i, s);
      }
    }
    if (!Simplex(2)) return numeric_limits<DOUBLE>::infinity();
    x = VD(n);
    for (int i = 0; i < m; i++) if (B[i] < n) x[B[i]] = D[i][n + 1];
    return D[m][n + 1];
  }
};

int main() {

  const int m = 4;
  const int n = 3;
  DOUBLE _A[m][n] = {
    { 6, -1, 0 },
    { -1, -5, 0 },
    { 1, 5, 1 },
    { -1, -5, -1 }
  };
  DOUBLE _b[m] = { 10, -4, 5, -5 };
  DOUBLE _c[n] = { 1, -1, 0 };

  VVD A(m);
  VD b(_b, _b + m);
  VD c(_c, _c + n);
  for (int i = 0; i < m; i++) A[i] = VD(_A[i], _A[i] + n);

  LPSolver solver(A, b, c);
  VD x;
  DOUBLE value = solver.Solve(x);

  cerr << "VALUE: " << value << endl; // VALUE: 1.29032
  cerr << "SOLUTION:"; // SOLUTION: 1.74194 0.451613 1
  for (size_t i = 0; i < x.size(); i++) cerr << " " << x[i];
  cerr << endl;
  return 0;
}
```
### SuffixArray.cc
```cpp
// Suffix array construction in O(L log^2 L) time.  Routine for
// computing the length of the longest common prefix of any two
// suffixes in O(log L) time.
//
// INPUT:   string s
//
// OUTPUT:  array suffix[] such that suffix[i] = index (from 0 to L-1)
//          of substring s[i...L-1] in the list of sorted suffixes.
//          That is, if we take the inverse of the permutation suffix[],
//          we get the actual suffix array.

#include <vector>
#include <iostream>
#include <string>

using namespace std;

struct SuffixArray {
  const int L;
  string s;
  vector<vector<int> > P;
  vector<pair<pair<int,int>,int> > M;

  SuffixArray(const string &s) : L(s.length()), s(s), P(1, vector<int>(L, 0)), M(L) {
    for (int i = 0; i < L; i++) P[0][i] = int(s[i]);
    for (int skip = 1, level = 1; skip < L; skip *= 2, level++) {
      P.push_back(vector<int>(L, 0));
      for (int i = 0; i < L; i++) 
	M[i] = make_pair(make_pair(P[level-1][i], i + skip < L ? P[level-1][i + skip] : -1000), i);
      sort(M.begin(), M.end());
      for (int i = 0; i < L; i++) 
	P[level][M[i].second] = (i > 0 && M[i].first == M[i-1].first) ? P[level][M[i-1].second] : i;
    }    
  }

  vector<int> GetSuffixArray() { return P.back(); }

  // returns the length of the longest common prefix of s[i...L-1] and s[j...L-1]
  int LongestCommonPrefix(int i, int j) {
    int len = 0;
    if (i == j) return L - i;
    for (int k = P.size() - 1; k >= 0 && i < L && j < L; k--) {
      if (P[k][i] == P[k][j]) {
	i += 1 << k;
	j += 1 << k;
	len += 1 << k;
      }
    }
    return len;
  }
};

// BEGIN CUT
// The following code solves UVA problem 11512: GATTACA.
#define TESTING
#ifdef TESTING
int main() {
  int T;
  cin >> T;
  for (int caseno = 0; caseno < T; caseno++) {
    string s;
    cin >> s;
    SuffixArray array(s);
    vector<int> v = array.GetSuffixArray();
    int bestlen = -1, bestpos = -1, bestcount = 0;
    for (int i = 0; i < s.length(); i++) {
      int len = 0, count = 0;
      for (int j = i+1; j < s.length(); j++) {
	int l = array.LongestCommonPrefix(i, j);
	if (l >= len) {
	  if (l > len) count = 2; else count++;
	  len = l;
	}
      }
      if (len > bestlen || len == bestlen && s.substr(bestpos, bestlen) > s.substr(i, len)) {
	bestlen = len;
	bestcount = count;
	bestpos = i;
      }
    }
    if (bestlen == 0) {
      cout << "No repetitions found!" << endl;
    } else {
      cout << s.substr(bestpos, bestlen) << " " << bestcount << endl;
    }
  }
}

#else
// END CUT
int main() {

  // bobocel is the 0'th suffix
  //  obocel is the 5'th suffix
  //   bocel is the 1'st suffix
  //    ocel is the 6'th suffix
  //     cel is the 2'nd suffix
  //      el is the 3'rd suffix
  //       l is the 4'th suffix
  SuffixArray suffix("bobocel");
  vector<int> v = suffix.GetSuffixArray();
  
  // Expected output: 0 5 1 6 2 3 4
  //                  2
  for (int i = 0; i < v.size(); i++) cout << v[i] << " ";
  cout << endl;
  cout << suffix.LongestCommonPrefix(0, 2) << endl;
}
// BEGIN CUT
#endif
// END CUT
```
### TopologicalSort.cc
```cpp
// This function uses performs a non-recursive topological sort.
//
// Running time: O(|V|^2).  If you use adjacency lists (vector<map<int> >),
//               the running time is reduced to O(|E|).
//
//   INPUT:   w[i][j] = 1 if i should come before j, 0 otherwise
//   OUTPUT:  a permutation of 0,...,n-1 (stored in a vector)
//            which represents an ordering of the nodes which
//            is consistent with w
//
// If no ordering is possible, false is returned.

#include <iostream>
#include <queue>
#include <cmath>
#include <vector>

using namespace std;

typedef double T;
typedef vector<T> VT;
typedef vector<VT> VVT;

typedef vector<int> VI;
typedef vector<VI> VVI;

bool TopologicalSort (const VVI &w, VI &order){
  int n = w.size();
  VI parents (n);
  queue<int> q;
  order.clear();
  
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++)
      if (w[j][i]) parents[i]++;
      if (parents[i] == 0) q.push (i);
  }
  
  while (q.size() > 0){
    int i = q.front();
    q.pop();
    order.push_back (i);
    for (int j = 0; j < n; j++) if (w[i][j]){
      parents[j]--;
      if (parents[j] == 0) q.push (j);
    }
  }
  
  return (order.size() == n);
}
```

### UnionFind.cc
```cpp
#include <iostream>
#include <vector>
using namespace std;
int find(vector<int> &C, int x) { return (C[x] == x) ? x : C[x] = find(C, C[x]); }
void merge(vector<int> &C, int x, int y) { C[find(C, x)] = find(C, y); }
int main()
{
	int n = 5;
	vector<int> C(n);
	for (int i = 0; i < n; i++) C[i] = i;
	merge(C, 0, 2);
	merge(C, 1, 0);
	merge(C, 3, 4);
	for (int i = 0; i < n; i++) cout << i << " " << find(C, i) << endl;
	return 0;
}
```

### Splay
```cpp
#include <cstdio>
#include <algorithm>
using namespace std;

const int N_MAX = 130010;
const int oo = 0x3f3f3f3f;
struct Node
{
  Node *ch[2], *pre;
  int val, size;
  bool isTurned;
} nodePool[N_MAX], *null, *root;

Node *allocNode(int val)
{
  static int freePos = 0;
  Node *x = &nodePool[freePos ++];
  x->val = val, x->isTurned = false;
  x->ch[0] = x->ch[1] = x->pre = null;
  x->size = 1;
  return x;
}

inline void update(Node *x)
{
  x->size = x->ch[0]->size + x->ch[1]->size + 1;
}

inline void makeTurned(Node *x)
{
  if(x == null)
    return;
  swap(x->ch[0], x->ch[1]);
  x->isTurned ^= 1;
}

inline void pushDown(Node *x)
{
  if(x->isTurned)
  {
    makeTurned(x->ch[0]);
    makeTurned(x->ch[1]);
    x->isTurned ^= 1;
  }
}

inline void rotate(Node *x, int c)
{
  Node *y = x->pre;
  x->pre = y->pre;
  if(y->pre != null)
    y->pre->ch[y == y->pre->ch[1]] = x;
  y->ch[!c] = x->ch[c];
  if(x->ch[c] != null)
    x->ch[c]->pre = y;
  x->ch[c] = y, y->pre = x;
  update(y);
  if(y == root)
    root = x;
}

void splay(Node *x, Node *p)
{
  while(x->pre != p)
  {
    if(x->pre->pre == p)
      rotate(x, x == x->pre->ch[0]);
    else
    {
      Node *y = x->pre, *z = y->pre;
      if(y == z->ch[0])
      {
        if(x == y->ch[0])
          rotate(y, 1), rotate(x, 1);
        else
          rotate(x, 0), rotate(x, 1);
      }
      else
      {
        if(x == y->ch[1])
          rotate(y, 0), rotate(x, 0);
        else
          rotate(x, 1), rotate(x, 0);
      }
    }
  }
  update(x);
}

void select(int k, Node *fa)
{
  Node *now = root;
  while(1)
  {
    pushDown(now);
    int tmp = now->ch[0]->size + 1;
    if(tmp == k)
      break;
    else if(tmp < k)
      now = now->ch[1], k -= tmp;
    else
      now = now->ch[0];
  }
  splay(now, fa);
}

Node *makeTree(Node *p, int l, int r)
{
  if(l > r)
    return null;
  int mid = (l + r) / 2;
  Node *x = allocNode(mid);
  x->pre = p;
  x->ch[0] = makeTree(x, l, mid - 1);
  x->ch[1] = makeTree(x, mid + 1, r);
  update(x);
  return x;
}

int main()
{
  int n, m;
  null = allocNode(0);
  null->size = 0;
  root = allocNode(0);
  root->ch[1] = allocNode(oo);
  root->ch[1]->pre = root;
  update(root);

  scanf("%d%d", &n, &m);
  root->ch[1]->ch[0] = makeTree(root->ch[1], 1, n);
  splay(root->ch[1]->ch[0], null);

  while(m --)
  {
    int a, b;
    scanf("%d%d", &a, &b);
    a ++, b ++;
    select(a - 1, null);
    select(b + 1, root);
    makeTurned(root->ch[1]->ch[0]);
  }

  for(int i = 1; i <= n; i ++)
  {
    select(i + 1, null);
    printf("%d ", root->val);
  }
}
```

## Java
### Dates.java
```java
// Example of using Java's built-in date calculation routines

import java.text.SimpleDateFormat;
import java.util.*;

public class Dates {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        SimpleDateFormat sdf = new SimpleDateFormat("M/d/yyyy");
        while (true) {
            int n = s.nextInt();
            if (n == 0) break;
            GregorianCalendar c = new GregorianCalendar(n, Calendar.JANUARY, 1);
            while (c.get(Calendar.DAY_OF_WEEK) != Calendar.SATURDAY) 
		c.add(Calendar.DAY_OF_YEAR, 1);
            for (int i = 0; i < 12; i++) {
                System.out.println(sdf.format(c.getTime()));
                while (c.get(Calendar.MONTH) == i) c.add(Calendar.DAY_OF_YEAR, 7);
            }
        }
    }
}
```
### DecFormat.java
```java
// examples for printing floating point numbers

import java.util.*;
import java.io.*;
import java.text.DecimalFormat;

public class DecFormat {
    public static void main(String[] args) {
        DecimalFormat fmt;

        // round to at most 2 digits, leave of digits if not needed
        fmt = new DecimalFormat("#.##");
        System.out.println(fmt.format(12345.6789)); // produces 12345.68
        System.out.println(fmt.format(12345.0)); // produces 12345
        System.out.println(fmt.format(0.0)); // produces 0
        System.out.println(fmt.format(0.01)); // produces .1

        // round to precisely 2 digits
        fmt = new DecimalFormat("#.00");
        System.out.println(fmt.format(12345.6789)); // produces 12345.68
        System.out.println(fmt.format(12345.0)); // produces 12345.00
        System.out.println(fmt.format(0.0)); // produces .00

        // round to precisely 2 digits, force leading zero
        fmt = new DecimalFormat("0.00");
        System.out.println(fmt.format(12345.6789)); // produces 12345.68
        System.out.println(fmt.format(12345.0)); // produces 12345.00
        System.out.println(fmt.format(0.0)); // produces 0.00

        // round to precisely 2 digits, force leading zeros
        fmt = new DecimalFormat("000000000.00");
        System.out.println(fmt.format(12345.6789)); // produces 000012345.68
        System.out.println(fmt.format(12345.0)); // produces 000012345.00
        System.out.println(fmt.format(0.0)); // produces 000000000.00

        // force leading '+'
        fmt = new DecimalFormat("+0;-0");
        System.out.println(fmt.format(12345.6789)); // produces +12346
        System.out.println(fmt.format(-12345.6789)); // produces -12346
        System.out.println(fmt.format(0)); // produces +0

        // force leading positive/negative, pad to 2
        fmt = new DecimalFormat("positive 00;negative 0");
        System.out.println(fmt.format(1)); // produces "positive 01"
        System.out.println(fmt.format(-1)); // produces "negative 01"

        // qoute special chars (#)
        fmt = new DecimalFormat("text with '#' followed by #");
        System.out.println(fmt.format(12.34)); // produces "text with # followed by 12"

        // always show "."
        fmt = new DecimalFormat("#.#");
        fmt.setDecimalSeparatorAlwaysShown(true);
        System.out.println(fmt.format(12.34)); // produces "12.3"
        System.out.println(fmt.format(12)); // produces "12."
        System.out.println(fmt.format(0.34)); // produces "0.3"

        // different grouping distances:
        fmt = new DecimalFormat("#,####.###");
        System.out.println(fmt.format(123456789.123)); // produces "1,2345,6789.123"

        // scientific:
        fmt = new DecimalFormat("0.000E00");
        System.out.println(fmt.format(123456789.123)); // produces "1.235E08"
        System.out.println(fmt.format(-0.000234)); // produces "-2.34E-04"

        // using variable number of digits:
        fmt = new DecimalFormat("0");
        System.out.println(fmt.format(123.123)); // produces "123"
        fmt.setMinimumFractionDigits(8);
        System.out.println(fmt.format(123.123)); // produces "123.12300000"
        fmt.setMaximumFractionDigits(0);
        System.out.println(fmt.format(123.123)); // produces "123"

        // note: to pad with spaces, you need to do it yourself:
        // String out = fmt.format(...)
        // while (out.length() < targlength) out = " "+out;
    }
}
```
### Geom3D.java
```java
public class Geom3D {
  // distance from point (x, y, z) to plane aX + bY + cZ + d = 0
  public static double ptPlaneDist(double x, double y, double z,
      double a, double b, double c, double d) {
    return Math.abs(a*x + b*y + c*z + d) / Math.sqrt(a*a + b*b + c*c);
  }
  
  // distance between parallel planes aX + bY + cZ + d1 = 0 and
  // aX + bY + cZ + d2 = 0
  public static double planePlaneDist(double a, double b, double c,
      double d1, double d2) {
    return Math.abs(d1 - d2) / Math.sqrt(a*a + b*b + c*c);
  }
  
  // distance from point (px, py, pz) to line (x1, y1, z1)-(x2, y2, z2)
  // (or ray, or segment; in the case of the ray, the endpoint is the
  // first point)
  public static final int LINE = 0;
  public static final int SEGMENT = 1;
  public static final int RAY = 2;
  public static double ptLineDistSq(double x1, double y1, double z1,
      double x2, double y2, double z2, double px, double py, double pz,
      int type) {
    double pd2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
    
    double x, y, z;
    if (pd2 == 0) {
      x = x1;
      y = y1;
      z = z1;
    } else {
      double u = ((px-x1)*(x2-x1) + (py-y1)*(y2-y1) + (pz-z1)*(z2-z1)) / pd2;
      x = x1 + u * (x2 - x1);
      y = y1 + u * (y2 - y1);
      z = z1 + u * (z2 - z1);
      if (type != LINE && u < 0) {
        x = x1;
        y = y1;
        z = z1;
      }
      if (type == SEGMENT && u > 1.0) {
        x = x2;
        y = y2;
        z = z2;
      }
    }
    
    return (x-px)*(x-px) + (y-py)*(y-py) + (z-pz)*(z-pz);
  }
  
  public static double ptLineDist(double x1, double y1, double z1,
      double x2, double y2, double z2, double px, double py, double pz,
      int type) {
    return Math.sqrt(ptLineDistSq(x1, y1, z1, x2, y2, z2, px, py, pz, type));
  }
}
```
### JavaGeometry.java
```java
// In this example, we read an input file containing three lines, each
// containing an even number of doubles, separated by commas.  The first two
// lines represent the coordinates of two polygons, given in counterclockwise 
// (or clockwise) order, which we will call "A" and "B".  The last line 
// contains a list of points, p[1], p[2], ...
//
// Our goal is to determine:
//   (1) whether B - A is a single closed shape (as opposed to multiple shapes)
//   (2) the area of B - A
//   (3) whether each p[i] is in the interior of B - A
//
// INPUT:
//   0 0 10 0 0 10
//   0 0 10 10 10 0
//   8 6
//   5 1
//
// OUTPUT:
//   The area is singular.
//   The area is 25.0
//   Point belongs to the area.
//   Point does not belong to the area.

import java.util.*;
import java.awt.geom.*;
import java.io.*;

public class JavaGeometry {

    // make an array of doubles from a string
    static double[] readPoints(String s) {
        String[] arr = s.trim().split("\\s++");
        double[] ret = new double[arr.length];
        for (int i = 0; i < arr.length; i++) ret[i] = Double.parseDouble(arr[i]);
        return ret;
    }

    // make an Area object from the coordinates of a polygon
    static Area makeArea(double[] pts) {
        Path2D.Double p = new Path2D.Double();
        p.moveTo(pts[0], pts[1]);
        for (int i = 2; i < pts.length; i += 2) p.lineTo(pts[i], pts[i+1]);
        p.closePath();
        return new Area(p);        
    }

    // compute area of polygon
    static double computePolygonArea(ArrayList<Point2D.Double> points) {
        Point2D.Double[] pts = points.toArray(new Point2D.Double[points.size()]);  
        double area = 0;
        for (int i = 0; i < pts.length; i++){
            int j = (i+1) % pts.length;
            area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
        }        
        return Math.abs(area)/2;
    }

    // compute the area of an Area object containing several disjoint polygons
    static double computeArea(Area area) {
        double totArea = 0;
        PathIterator iter = area.getPathIterator(null);
        ArrayList<Point2D.Double> points = new ArrayList<Point2D.Double>();

        while (!iter.isDone()) {
            double[] buffer = new double[6];
            switch (iter.currentSegment(buffer)) {
            case PathIterator.SEG_MOVETO:
            case PathIterator.SEG_LINETO:
                points.add(new Point2D.Double(buffer[0], buffer[1]));
                break;
            case PathIterator.SEG_CLOSE:
                totArea += computePolygonArea(points);
                points.clear();
                break;
            }
            iter.next();
        }
        return totArea;
    }

    // notice that the main() throws an Exception -- necessary to
    // avoid wrapping the Scanner object for file reading in a 
    // try { ... } catch block.
    public static void main(String args[]) throws Exception {

        Scanner scanner = new Scanner(new File("input.txt"));
        // also,
        //   Scanner scanner = new Scanner (System.in);

        double[] pointsA = readPoints(scanner.nextLine());
        double[] pointsB = readPoints(scanner.nextLine());
        Area areaA = makeArea(pointsA);
        Area areaB = makeArea(pointsB);
        areaB.subtract(areaA);
        // also,
        //   areaB.exclusiveOr (areaA);
        //   areaB.add (areaA);
        //   areaB.intersect (areaA);
        
        // (1) determine whether B - A is a single closed shape (as 
        //     opposed to multiple shapes)
        boolean isSingle = areaB.isSingular();
        // also,
        //   areaB.isEmpty();

        if (isSingle)
            System.out.println("The area is singular.");
        else
            System.out.println("The area is not singular.");
        
        // (2) compute the area of B - A
        System.out.println("The area is " + computeArea(areaB) + ".");
        
        // (3) determine whether each p[i] is in the interior of B - A
        while (scanner.hasNextDouble()) {
            double x = scanner.nextDouble();
            assert(scanner.hasNextDouble());
            double y = scanner.nextDouble();

            if (areaB.contains(x,y)) {
                System.out.println ("Point belongs to the area.");
            } else {
                System.out.println ("Point does not belong to the area.");
            }
        }

        // Finally, some useful things we didn't use in this example:
        //
        //   Ellipse2D.Double ellipse = new Ellipse2D.Double (double x, double y, 
        //                                                    double w, double h);
        //
        //     creates an ellipse inscribed in box with bottom-left corner (x,y)
        //     and upper-right corner (x+y,w+h)
        // 
        //   Rectangle2D.Double rect = new Rectangle2D.Double (double x, double y, 
        //                                                     double w, double h);
        //
        //     creates a box with bottom-left corner (x,y) and upper-right 
        //     corner (x+y,w+h)
        //
        // Each of these can be embedded in an Area object (e.g., new Area (rect)).

    }
}
```
### LogLan.java
```java
// Code which demonstrates the use of Java's regular expression libraries.
// This is a solution for 
//
//   Loglan: a logical language
//   http://acm.uva.es/p/v1/134.html
//
// In this problem, we are given a regular language, whose rules can be
// inferred directly from the code.  For each sentence in the input, we must
// determine whether the sentence matches the regular expression or not.  The
// code consists of (1) building the regular expression (which is fairly
// complex) and (2) using the regex to match sentences.

import java.util.*;
import java.util.regex.*;

public class LogLan {

    public static String BuildRegex (){
	String space = " +";

	String A = "([aeiou])";
	String C = "([a-z&&[^aeiou]])";
	String MOD = "(g" + A + ")";
	String BA = "(b" + A + ")";
	String DA = "(d" + A + ")";
	String LA = "(l" + A + ")";
	String NAM = "([a-z]*" + C + ")";
	String PREDA = "(" + C + C + A + C + A + "|" + C + A + C + C + A + ")";

	String predstring = "(" + PREDA + "(" + space + PREDA + ")*)";
	String predname = "(" + LA + space + predstring + "|" + NAM + ")";
	String preds = "(" + predstring + "(" + space + A + space + predstring + ")*)";
	String predclaim = "(" + predname + space + BA + space + preds + "|" + DA + space +
            preds + ")";
	String verbpred = "(" + MOD + space + predstring + ")";
	String statement = "(" + predname + space + verbpred + space + predname + "|" + 
            predname + space + verbpred + ")";
	String sentence = "(" + statement + "|" + predclaim + ")";

	return "^" + sentence + "$";
    }

    public static void main (String args[]){

	String regex = BuildRegex();
	Pattern pattern = Pattern.compile (regex);
	
	Scanner s = new Scanner(System.in);
	while (true) {

            // In this problem, each sentence consists of multiple lines, where the last 
	    // line is terminated by a period.  The code below reads lines until
	    // encountering a line whose final character is a '.'.  Note the use of
            //
            //    s.length() to get length of string
            //    s.charAt() to extract characters from a Java string
            //    s.trim() to remove whitespace from the beginning and end of Java string
            //
            // Other useful String manipulation methods include
            //
            //    s.compareTo(t) < 0 if s < t, lexicographically
            //    s.indexOf("apple") returns index of first occurrence of "apple" in s
            //    s.lastIndexOf("apple") returns index of last occurrence of "apple" in s
            //    s.replace(c,d) replaces occurrences of character c with d
            //    s.startsWith("apple) returns (s.indexOf("apple") == 0)
            //    s.toLowerCase() / s.toUpperCase() returns a new lower/uppercased string
            //
            //    Integer.parseInt(s) converts s to an integer (32-bit)
            //    Long.parseLong(s) converts s to a long (64-bit)
            //    Double.parseDouble(s) converts s to a double
            
	    String sentence = "";
	    while (true){
		sentence = (sentence + " " + s.nextLine()).trim();
		if (sentence.equals("#")) return;
		if (sentence.charAt(sentence.length()-1) == '.') break;		
	    }

            // now, we remove the period, and match the regular expression

            String removed_period = sentence.substring(0, sentence.length()-1).trim();
	    if (pattern.matcher (removed_period).find()){
		System.out.println ("Good");
	    } else {
		System.out.println ("Bad!");
	    }
	}
    }
}
```
### MaxFlow.java
```java
// Fattest path network flow algorithm using an adjacency matrix.
//
// Running time: O(|E|^2 log (|V| * U), where U is the largest
//               capacity of any edge.  If you replace the 'fattest
//               path' search with a minimum number of edges search,
//               the running time becomes O(|E|^2 |V|).
//
// INPUT: cap -- a matrix such that cap[i][j] is the capacity of
//               a directed edge from node i to node j
//
//               * Note that it is legitimate to create an i->j
//                 edge without a corresponding j->i edge.
//
//               * Note that for an undirected edge, set
//                 both cap[i][j] and cap[j][i] to the capacity of
//                 the undirected edge.
//
//        source -- starting node
//        sink -- ending node
//
// OUTPUT: value of maximum flow; also, the flow[][] matrix will
//         contain both positive and negative integers -- if you
//         want the actual flow assignments, look at the
//         *positive* flow values only.
//
// To use this, create a MaxFlow object, and call it like this:
//
//   MaxFlow nf;
//   int maxflow = nf.getMaxFlow(cap,source,sink);

import java.util.*;

public class MaxFlow {
    boolean found[];
    int N, cap[][], flow[][], dad[], dist[];

    boolean searchFattest(int source, int sink) {
	Arrays.fill(found, false);
	Arrays.fill(dist, 0);
	dist[source] = Integer.MAX_VALUE / 2;
        while (source != N) {
            int best = N;
            found[source] = true;
            if (source == sink) break;
            for (int k = 0; k < N; k++) {
		if (found[k]) continue;
		int possible = Math.min(cap[source][k] - flow[source][k], dist[source]);
		if (dist[k] < possible) {
		    dist[k] = possible;
		    dad[k] = source; 
		}
                if (dist[k] > dist[best]) best = k;
	    }
            source = best;
        }
        return found[sink];
    }

    boolean searchShortest(int source, int sink) {
	Arrays.fill(found, false);
	Arrays.fill(dist, Integer.MAX_VALUE/2);
	dist[source] = 0;
        while (source != N) {
            int best = N;
            found[source] = true;
            if (source == sink) break;
            for (int k = 0; k < N; k++) {
		if (found[k]) continue;
                if (cap[source][k] - flow[source][k] > 0) {
                    if (dist[k] > dist[source] + 1){
                        dist[k] = dist[source] + 1;
                        dad[k] = source;
                    }
                }
                if (dist[k] < dist[best]) best = k;
	    }
            source = best;
        }
        return found[sink];
    }

    public int getMaxFlow(int cap[][], int source, int sink) {
        this.cap = cap;
        N = cap.length;
        found = new boolean[N];
        flow = new int[N][N];
        dist = new int[N+1];
        dad = new int[N];
    
        int totflow = 0;
        while (searchFattest(source, sink)) {
            int amt = Integer.MAX_VALUE;
            for (int x = sink; x != source; x = dad[x])
                amt = Math.min(amt, cap[dad[x]][x] - flow[dad[x]][x]);
            for (int x = sink; x != source; x = dad[x]) {
                flow[dad[x]][x] += amt;
                flow[x][dad[x]] -= amt;
            }
            totflow += amt;
        }

        return totflow;
    }
  
    public static void main(String args[]) {
	MaxFlow flow = new MaxFlow();
	int cap[][] = {{0, 3, 4, 5, 0},
		       {0, 0, 2, 0, 0},
		       {0, 0, 0, 4, 1},
		       {0, 0, 0, 0, 10},
		       {0 ,0, 0, 0, 0}};

	// should print out "10"

	System.out.println(flow.getMaxFlow(cap, 0, 4));
    }
}
```
### MinCostMaxFlow.java
```java
// Min cost max flow algorithm using an adjacency matrix.  If you
// want just regular max flow, setting all edge costs to 1 gives
// running time O(|E|^2 |V|).
//
// Running time: O(min(|V|^2 * totflow, |V|^3 * totcost))
//
// INPUT: cap -- a matrix such that cap[i][j] is the capacity of
//               a directed edge from node i to node j
//
//        cost -- a matrix such that cost[i][j] is the (positive)
//                cost of sending one unit of flow along a 
//                directed edge from node i to node j
//
//        source -- starting node
//        sink -- ending node
//
// OUTPUT: max flow and min cost; the matrix flow will contain
//         the actual flow values (note that unlike in the MaxFlow
//         code, you don't need to ignore negative flow values -- there
//         shouldn't be any)
//
// To use this, create a MinCostMaxFlow object, and call it like this:
//
//   MinCostMaxFlow nf;
//   int maxflow = nf.getMaxFlow(cap,cost,source,sink);

import java.util.*;

public class MinCostMaxFlow {
    boolean found[];
    int N, cap[][], flow[][], cost[][], dad[], dist[], pi[];
    
    static final int INF = Integer.MAX_VALUE / 2 - 1;
    
    boolean search(int source, int sink) {
	Arrays.fill(found, false);
	Arrays.fill(dist, INF);
	dist[source] = 0;

	while (source != N) {
	    int best = N;
	    found[source] = true;
	    for (int k = 0; k < N; k++) {
		if (found[k]) continue;
		if (flow[k][source] != 0) {
		    int val = dist[source] + pi[source] - pi[k] - cost[k][source];
		    if (dist[k] > val) {
			dist[k] = val;
			dad[k] = source;
		    }
		}
		if (flow[source][k] < cap[source][k]) {
		    int val = dist[source] + pi[source] - pi[k] + cost[source][k];
		    if (dist[k] > val) {
			dist[k] = val;
			dad[k] = source;
		    }
		}
		
		if (dist[k] < dist[best]) best = k;
	    }
	    source = best;
	}
	for (int k = 0; k < N; k++)
	    pi[k] = Math.min(pi[k] + dist[k], INF);
	return found[sink];
    }
    
    
    int[] getMaxFlow(int cap[][], int cost[][], int source, int sink) {
	this.cap = cap;
	this.cost = cost;
	
	N = cap.length;
        found = new boolean[N];
        flow = new int[N][N];
        dist = new int[N+1];
        dad = new int[N];
        pi = new int[N];
	
	int totflow = 0, totcost = 0;
	while (search(source, sink)) {
	    int amt = INF;
	    for (int x = sink; x != source; x = dad[x])
		amt = Math.min(amt, flow[x][dad[x]] != 0 ? flow[x][dad[x]] :
                       cap[dad[x]][x] - flow[dad[x]][x]);
	    for (int x = sink; x != source; x = dad[x]) {
		if (flow[x][dad[x]] != 0) {
		    flow[x][dad[x]] -= amt;
		    totcost -= amt * cost[x][dad[x]];
		} else {
		    flow[dad[x]][x] += amt;
		    totcost += amt * cost[dad[x]][x];
		}
	    }
	    totflow += amt;
	}
	
	return new int[]{ totflow, totcost };
    }
  
    public static void main (String args[]){
        MinCostMaxFlow flow = new MinCostMaxFlow();
        int cap[][] = {{0, 3, 4, 5, 0},
                       {0, 0, 2, 0, 0},
                       {0, 0, 0, 4, 1},
                       {0, 0, 0, 0, 10},
                       {0, 0, 0, 0, 0}};

        int cost1[][] = {{0, 1, 0, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0}};

        int cost2[][] = {{0, 0, 1, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0}};
        
        // should print out:
        //   10 1
        //   10 3

        int ret1[] = flow.getMaxFlow(cap, cost1, 0, 4);
        int ret2[] = flow.getMaxFlow(cap, cost2, 0, 4);
        
        System.out.println (ret1[0] + " " + ret1[1]);
        System.out.println (ret2[0] + " " + ret2[1]);
    }
}
```
### SegmentTreeLazy.java
```java
public class SegmentTreeRangeUpdate {
	public long[] leaf;
	public long[] update;
	public int origSize;
	public SegmentTreeRangeUpdate(int[] list)	{
		origSize = list.length;
		leaf = new long[4*list.length];
		update = new long[4*list.length];
		build(1,0,list.length-1,list);
	}
	public void build(int curr, int begin, int end, int[] list)	{
		if(begin == end)
			leaf[curr] = list[begin];
		else	{
			int mid = (begin+end)/2;
			build(2 * curr, begin, mid, list);
			build(2 * curr + 1, mid+1, end, list);
			leaf[curr] = leaf[2*curr] + leaf[2*curr+1];
		}
	}
	public void update(int begin, int end, int val)	{
		update(1,0,origSize-1,begin,end,val);
	}
	public void update(int curr,  int tBegin, int tEnd, int begin, int end, int val)	{
		if(tBegin >= begin && tEnd <= end)
			update[curr] += val;
		else	{
			leaf[curr] += (Math.min(end,tEnd)-Math.max(begin,tBegin)+1) * val;
			int mid = (tBegin+tEnd)/2;
			if(mid >= begin && tBegin <= end)
				update(2*curr, tBegin, mid, begin, end, val);
			if(tEnd >= begin && mid+1 <= end)
				update(2*curr+1, mid+1, tEnd, begin, end, val);
		}
	}
	public long query(int begin, int end)	{
		return query(1,0,origSize-1,begin,end);
	}
	public long query(int curr, int tBegin, int tEnd, int begin, int end)	{
		if(tBegin >= begin && tEnd <= end)	{
			if(update[curr] != 0)	{
				leaf[curr] += (tEnd-tBegin+1) * update[curr];
				if(2*curr < update.length){
					update[2*curr] += update[curr];
					update[2*curr+1] += update[curr];
				}
				update[curr] = 0;
			}
			return leaf[curr];
		}
		else	{
			leaf[curr] += (tEnd-tBegin+1) * update[curr];
			if(2*curr < update.length){
				update[2*curr] += update[curr];
				update[2*curr+1] += update[curr];
			}
			update[curr] = 0;
			int mid = (tBegin+tEnd)/2;
			long ret = 0;
			if(mid >= begin && tBegin <= end)
				ret += query(2*curr, tBegin, mid, begin, end);
			if(tEnd >= begin && mid+1 <= end)
				ret += query(2*curr+1, mid+1, tEnd, begin, end);
			return ret;
		}
	}
}
```

From Past ACM Cheat Sheet 2014:
### Bitwise Operators 

| operation | syntax |
| --- | --- |
| bitwise AND | `&` |
| bitwise OR | `|` |
| bitwise XOR | `^` |
| left shift | `<<` |
| right shift | `>>` |
| complement | `~` |
| all 1’s in binary | `-1` |

##### Macro to check if a bit is set
```cpp
#define CHECK_BIT(variable, position) ((variable) & (1 << (position)))
```
and use it like this
```cpp
CHECK_BIT(temp, 3)
```
### Inversions
```cpp
// Count the collisions problem
// Input:  RRRLLL
// Output: 9
#include<iostream>
using namespace std;
int main() {
    int count_r,total;
    string input;
    while(getline(cin,input)) {
        count_r = total = 0;
        for(int i=0; i < input.length();++i)
            if(input[i] == 'R') count_r++;
            else total += count_r;
    cout << total << endl;
    }
    return 0;
}
```
### Mathematics
#### Algebra 
##### Sum of Powers

##### Fast Exponentiation
```cpp 
// This is a very good way to reduce the overhead of the <cmath> library pow function   
double pow(double a, int n) {
    if(n == 0) return 1;
    if(n == 1) return a;
    double t = pow(a, n/2);
    return t * t * pow(a, n%2);
}
```
##### Greatest Common Divisor (GCD)           
```cpp
int gcd(int a, int b) {
    while(b){int r = a % b; a = b; b = r;} 
    return a;    
}
```

##### Euclidian Algorithm
```cpp
while (a > 0 && b > 0)
a > b ? a = a - (2*b); : b = b - (2*a);
a > b ? return a; : return b;
```
### Primes
##### Sieve of Eratosthenes
```cpp
#include<vector>
#include<cmath>
#include<iostream>
using namespace std;

vector<bool> num;

void sieve(int n) {
  num[0]=0;
  num[1]=0;
  int m=(int)sqrt(n);
  for (int i=2; i<=m; i++) 
    if (num[i])
      for (int k=i*i; k<=n; k+=i) 
        num[k]=0;
}
int main() {
    num = vector<bool>(1000000,1);
    sieve(1000000);
    for(int i=0;i<num.size();++i) 
        if(num[i]) cout<<i<<endl;
}
```

##### O(sqrt(x)) Exhaustive Primality Test

```cpp
// Use the Sieve for numbers 1-1000000, 
// but anything larger than 1000000, use IsPrimeSlow
#include <cmath>
#define EPS 1e-7
typedef long long LL;
bool IsPrimeSlow (LL x)
{
  if(x<=1) return false;
  if(x<=3) return true;
  if (!(x%2) || !(x%3)) return false;
  LL s=(LL)(sqrt((double)(x))+EPS);
  for(LL i=5;i<=s;i+=6)
  {
    if (!(x%i) || !(x%(i+2))) return false;
  }
  return true;
}
```

## Handling Input
### Eating Newline Characters Before `getline()`

##### Input:
```
3
some string with spaces
another string with spaces
third string with spaces
```

##### Output:
```
some string with spaces
another string with spaces
third string with spaces
```

##### solution 1:
```cpp
#include<iostream>
//#include <sstream>
using namespace std;
int main() {
  int N;
  string s;
//getline(cin, s);
//stringstream ss (s); 
  cin >> N;
//ss >> N;
  cin.get(); // get the newline character that is after the number 3,
 // otherwise getline will get an empty string the first time
  while(N--) {
    getline(cin, s);
    cout << s << endl;
  }
  return 0;
}
```

##### solution 2:
```cpp
#include<iostream>
#include <sstream>
using namespace std;
int main() {
  int N;
  string s;
  getline(cin, s);
  stringstream ss (s); 
  ss >> N;
  while(N--) {
    getline(cin, s);
    cout << s << endl;
  }
  return 0;
}
```
### Set stream to not ignore whitespace

##### Input:
```
a b c d
```

##### Output:
```
0 1 2 3 4 5 6 7
```

##### Solution 1 (faster):
```cpp
#include<iostream>
using namespace std;
int main(){
  char c;
  for(int i=0; cin>>noskipws>>c ;++i)
    cout << i << “ “;
  return 0;
}
```

##### Solution 2:
```cpp
#include<iostream>
using namespace std;
int main(){
   string s;
   getline(cin,s);  
   for(int i=0; i < string.length() ;++i)
     cout << i << “ “;
   return 0;
}
```
### `toBase()`
```cpp
// Assumes “int number” parameter is base10
string toBase(int number, int base) {
  string vals = "0123456789ABCDEFGHIJLMNOP";
  string result = "";
  while(number) {
    result = vals[number%base] + result;
    number /= base;
  }
  return result;
}
```

### `isPalindrome()`
```
bool isPalindrome(string s) {
  for(int i=0,max=s.length()/2,len=s.length()-1;i<max;++i)
    if(s[i]!=s[len-i]) return false;
  return true;
}
```

### Finding All The Armstrong Numbers Up To 1000

```
/*
Problem: Find all Armstrong numbers up to 1000

An Armstrong number is a number that is the sum of its
own digits, each raise to the power of the number of
digits.

Steps:

1. Begin with an n-digit number
2. Raise each digit to the nth power and compute the sum
3. If the sum is same as the n-digit number, it is an Armstrong number (i.e. for 3-digit number 153: 1^3 + 5^3 + 3^3 = 153)
4. Continue loop and test the next number
*/
```
```cpp
#include <iostream>
using namespace std;

bool isArmstrong(int x);
int pow(int n, int power);
int digits(long int n);

int main() 
{
    int min = 0, max = 1000;
    cout << "Program to find Armstrong numbers..." 
         << endl;
    cout << "Enter a range for the Armstrong number..." 
         << endl;
    cout << "Enter lowest value to test: ";
    cin >> min;
    cout << "Enter highest value to test: ";
    cin >> max;
    cout << endl;

    for (int i = min; i <= max; ++i)
        if (isArmstrong(i)) 
            cout << i << " ";

    cout << endl;
    return 0;
}

bool isArmstrong(int x)
{
    int n = x;
    int d = digits(x);
    int y = 0, z = x;

    while (z > 0)
    {
        x = z % 10;
        y = y + pow(x, d);
        z /= 10;
    }

    if (y == n) return true;
    else return false;
}

int pow(int n, int power)
{
    if (power == 1) return n;
    else return n * pow(n, power - 1);
}

int digits(long int n)
{
    if (n < 10) return 1;
    else return 1 + digits(n / 10);
}
```
```
/*
OUTPUT

Program to find Armstrong numbers...
Enter a range for the Armstrong number...
Enter lowest value to test: 0
Enter highest value to test: 1000

0 1 2 3 4 5 6 7 8 9 153 370 371 407
*/
```

### Finding Factorial Iteratively and Recursively
```
/*
The factorial of a non-negative integer 'n', denoted 
by n!, is the product of all positive integers less 
than or equal to n.
For example: 5! = 5 x 4 x 3 x 2 x 1 = 120
*/
```

```cpp
#include <iostream>
using namespace std;

int recursive_factorial(int n)
{
    if (n == 0) return 1;
    else return (n * recursive_factorial(n - 1));
}

int iterative_factorial(int n)
{
    int f = 1;
    for (int i = 1; i <= n; ++i)
        f *= i;

    return f;
}

int main()
{
    cout << "Find factorial of? ";
    int n; cin >> n;
    cout << "Using recursion, factorial is ";
    cout << recursive_factorial(n) << endl;
    cout << "Using iteration, factorial is ";
    cout << iterative_factorial(n) << endl;
    return 0;
}
```
```
/* OUTPUT

Find factorial of? 7
Using recursion, factorial is 5040
Using iteration, factorial is 5040
*/
```
### Finding GCD Recursively
```
/*
The greatest common divisor (gcd) also known as the
greatest common factor (gcf) or highest common factor
(hcf) of two or more non-zero integers, is the largest
positive integer that divides the numbers without a 
remainder.

Steps:

1. Accept two values m and n, whose GCD we want to find  
2. Determine the smaller value between m and n and assign it to d  
3. Divide both m and n by d, if the remainder in both the cases is 0 then d is the required GCD, print the value and exit.  
4. Else if either of the division produces a non-zero remainder, decrement d  
5. Repeat step 3-4 until a GCD is found.  
*/
```
```cpp
#include <iostream>
using namespace std;

int gcd(int m, int n, int d = -1)
{
    if (d == -1)
        d = m > n ? n : m;

    if (m % d == 0 && n % d == 0)
        return d;

    else return gcd (m, n, d - 1);
}

int main()
{
    int m, n;
    cout << "Enter first number: ";
    cin >> m;
    cout << "Enter second number: ";
    cin >> n;
    cout << "GCD is " << gcd(m, n);
    return 0;
}
```
```
/* OUTPUT

Enter first number: 56
Enter second number: 42
GCD is 14
*/
```
### Determining if a Number is Prime Iteratively 
```
/*
A prime number (or prime) is a natural number greater
than 1 that has no positive divisors other than 1 and
itself. A natural number greater than 1 that is not a
prime number is called a composite number.
Steps:

1. Get value n, which we want to test  
2. Let i = n - 1  
3. Divide n by i, if the remainder is non-zero the number is not prime, exit  
4. Repeat step 3 until we reach 1, if all division upto this point resulted in zero remainder, the number is prime.  
*/
```
```cpp
#include <iostream>
#include <cmath>
using namespace std;

int main()
{
    cout << "Program to test primality" << endl;
    cout << "Enter a number: ";
    int n = 0; cin >> n;

    // Loop from sqrt(n) to 1
    for (int i = sqrt(n); i > 0; --i)
    {
        if (i == 1)
        {
            cout << "Prime";
            break;
        }
        if ((n % i) == 0)
        {
            cout << "Not Prime";
            break;
        }
    }
    return 0;
}
```
```
/* OUTPUT

Program to test primality
Enter a number: 6
Not Prime
*/
```
### Finding the Largest Palindromic Number Formed by Multiplying Two 3-Digit Numbers.
```
/*
Problem: Find the largest palindromic number formed by
multiplying two 3-digit numbers.

A palindromic number or numeral palindrome is a number
that remains the same when its digits are reversed.

Steps:

1. Start with largest value of palindrome as 0  
2. Make a reverse copy of the number to test  
3. Compare it with the original number value  
4. If the two are exactly equal, the number is a palindrome  
5. If the number is a palindrome and is larger than the current stored largest value, store the number as the new largest  
6. Repeat steps 2-5 for all combinations of i * j, where i and j range from 100-999  
7. The largest value at the end is the required largest value.  
*/
```
```cpp
#include <iostream>
using namespace std;
bool isPalindrome(int number)
{
    int original = number;
    int reverse = 0;

    while(number)
    {
        int remain = number % 10;
        number /= 10;
        reverse = reverse * 10 + remain;
    }
    return reverse == original;
}

----- OR -------

#include <iostream>
#include <sstream>
using namespace std;
bool isPalindromeNum(int number)
{
    stringstream ss(number);
    string original;
    ss >> original;
    for(int i=0,max=s.length()/2,len=s.length()-1;i<max;++i)
        if(s[i]!=s[len-i]) return false;
    return true;
}
-----------------------
int main()
{
    long largest = 0;

    for (int i = 999; i > 99; --i)
        for (int j = 999; j > 99; --j)
            if (isPalindrome(i * j) && i * j > largest)
                    largest = i * j;

    cout << "Largest palindrome is " << largest;
    return 0;
}
```
```
/* OUTPUT

Largest palindrome is 906609
*/
```
### Finding the Largest Prime Factor of 600,851,475,143
```
/*
The prime factors of a positive integer are the prime
numbers that divide that integer exactly, without 
leaving a remainder. The process of finding these 
numbers is called integer factorization, or prime
factorization.
Steps:

1. Let n be the number whose prime factor is to be calculated  
2. Let s be the square root of n  
3. Divide n by s, if it leaves a remainder equal to zero, s is the prime factor, else continue to step 4  
4. Decrement by 1, repeat step 3 until a prime factor is found.  
*/
```
```cpp
#include <iostream>
#include <cmath>
using namespace std;

bool isPrime(long n, long i)
{
    if (i >= n) 
        return true;
    else if (n % i == 0) 
        return false;
    else return isPrime(n, ++i);
}

int main()
{
    long long n = 600851475143LL;
    long s = (long) sqrt(n);

    cout << "What is the largest prime factor "
         << "of the number 600851475143?" << endl;

    for (long i = s; i > 1; --i)
    {
        if ((n % i) == 0)
        {
            if (!isPrime(i, 2))
                continue;
            cout << "Answer: " << i;
            break;
        }
    }
    return 0;
}
```
```
/* OUTPUT

What is the largest prime factor of 600851475143?
Answer: 6857
*/
```
### Finding the Transpose of a Matrix with Multidimensional Arrays
```
/*
Problem: To find the transpose of a given matrix.

The transpose of a matrix is formed by turning all the
rows of a given matrix into columns and vice-versa. The
transpose of matrix A is written A^T.

Steps:
1. Get the values for matrix A with 'r' rows and 'c' columns  
2. Copy value at rth row and cth column in matrix A to the cth row and rth column of result matrix (i.e. R[c][r] = A[r][c])  
3. Repeat step 2 for each element of matrix A  
*/
```
```cpp
#include <iostream>
using namespace std;

int main()
{
    int r = 0, c = 0;
    cout << "Enter the size of the matrix..." << endl;
    cout << "How many rows? ";
    cin >> r;
    cout << "How many columns? ";
    cin >> c;

    const int rows = r;
    const int cols = c;
       
    int** matrix = new int*[rows];
    int** result = new int*[cols];

    int i = 0, j = 0;

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            matrix[r] = new int[cols];
            result[c] = new int[rows];
        }
    }

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            cout << "Enter value for Matrix[" << r + 1
                 << "  " << c + 1 << "]: ";
            cin >> matrix[r][c];
        }
    }
 
    for (r = 0; r < cols; ++r)
        for (c = 0; c < rows; ++c)
            result[r][c] = matrix[c][r];

    cout << endl << "Original matrix is..." << endl;
    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
            cout << result[c][r] << " ";
        cout << endl;
    }

    cout << endl << "Result of transpose..." << endl;
    for (c = 0; c < cols; ++c)
    {
        for (r = 0; r < rows; ++r)
            cout << result[c][r] << " ";
        cout << endl;
    }

    // Clean up
    for (r = 0; r < rows; ++r)
    {
        delete [] matrix[r];
        delete [] result[r];
    }
    delete [] matrix;
    delete [] result;

    return 0;
}
```
```
/* OUTPUT

Enter the size of the matrix...

How many rows? 3
How many columns? 2
Enter value for Matrix[1  1]: 1
Enter value for Matrix[1  2]: 2
Enter value for Matrix[2  1]: 3
Enter value for Matrix[2  2]: 4
Enter value for Matrix[3  1]: 5
Enter value for Matrix[3  2]: 6

Original matrix is...
1 2
3 4
5 6

Result of transpose...
1 3 5
2 4 6
*/
```
### Finding the Transpose of a Matrix with a Single Dimensional Array
```
/*
Problem: To find the transpose of a given matrix.

The transpose of a matrix is formed by turning all the
rows of a given matrix into columns and vice-versa. The
transpose of matrix A is written A^T.

Steps:

1. Get the values for matrix A with 'r' rows and 'c' columns  
2. Copy value at rth row and cth column in matrix A to the cth row and rth column of result matrix (i.e. R[c][r] = A[r][c])  
3. Repeat step 2 for each element of matrix A  
*/
```
```cpp
#include <iostream>
using namespace std;

int main()
{
    int r = 0, c = 0;
    cout << "Enter the size of the matrix..." << endl;
    cout << "How many rows? ";
    cin >> r;
    cout << "How many columns? ";
    cin >> c;

    const int rows = r;
    const int cols = c;

    // Use one large block of memory for the 2D array
    // matrix[i * cols + j] == matrix[i][j] 
    int* matrix = new int[rows * cols];
    int* result = new int[cols * rows];

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            cout << "Enter value for Matrix[" << r + 1
                 << "  " << c + 1 << "]: ";
            cin >> matrix[r * cols + c];
        }
    }    

    for (r = 0; r < cols; ++r)
        for (c = 0; c < rows; ++c)
            result[r * cols + c] = matrix[c * rows + r];

    cout << endl << "Original matrix is..." << endl;
    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
            cout << result[c * rows + r] << " ";
        cout << endl;
    }

    cout << endl << "Result of transpose..." << endl;
    for (c = 0; c < cols; ++c)
    {
        for (r = 0; r < rows; ++r)
            cout << result[c * rows + r] << " ";
        cout << endl;
    }

    // Clean up
    delete [] matrix;
    delete [] result;
    return 0;
}
```
```
/* OUTPUT

Enter the size of the matrix...

How many rows? 3
How many columns? 2
Enter value for Matrix[1  1]: 1
Enter value for Matrix[1  2]: 2
Enter value for Matrix[2  1]: 3
Enter value for Matrix[2  2]: 4
Enter value for Matrix[3  1]: 5
Enter value for Matrix[3  2]: 6

Original matrix is...
1 2
3 4
5 6

Result of transpose...
1 3 5
2 4 6
*/
```
### Iterative Binary Search
```
/*
Problem: Implement an Iterative Binary Search.

On average for finding any value in an unsorted array, 
complexity is proportional to the length of the array.

The situation changes significantly when the array is 
sorted. If we know it, random access capability can be
utilized very efficiently to quickly find the searched
value. The cost of the searching algorithm reduces to
binary logarithm of the array length. For reference,
log2(1000000) is approximately 20. It means that in the
worst case, the algorithm takes 20 steps to find a 
value in a sorted array of one million elements.

Steps:

1. Get the middle element  
2. If the middle element equals the searched value, the algorithm stops  
3. Otherwise, two cases are possible:  
    1. Searched value is less than the middle element. In this case go to step 1 for the part of the array before the middle element.  
    2. Searched value is greater than the middle element. In this case go to step 1 for the part of the array after the middle element.  
4. Iteration should stop when the searched element is found or when the sub-array has no elements. In the second case, we can conclude that the searched value is not present in the array.  
*/
```
```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int binarySearch(int haystack[], int needle, int length)
{
    int low = 0; 
    int high = length;
    int mid = (low + high) / 2;

    while (high >= low)
    {
        if (haystack[mid] == needle)
            return mid;
        else
        {
            if (needle > haystack[mid])
                low = mid + 1;
            else high = mid - 1;
            mid = (low + high) / 2;
        }
    }
    return -1; // not found
}

int main()
{    
    // Replace haystack with user input if needed
    // Haystack MUST BE SORTED
    int haystack[10] = {10, 11, 20, 22, 30,
                        33, 40, 44, 50, 55};
    int length = sizeof(haystack) / sizeof(int);
    srand(time(NULL));
    int needle = haystack[rand() % 10];

    cout << "This is the array: ";
    for (int i = 0; i < length; ++i)
        cout << haystack[i] << " ";

    cout << endl;
    cout << "Searching for " << needle << endl;
    cout << "Value is at array index: ";
    cout << binarySearch(haystack, needle, length);
    return 0;
}
```
```
/* OUTPUT

This is the array: 10 11 20 22 30 33 40 44 50 55
Searching for 40
Value is at array index: 6
*/
```
Removing Vowels from a String
```cpp
#include <iostream>
#include <string>
#include <cctype>
using namespace std;

const string vowels = "aeiou";

bool isVowel(char chr)
{
    for (int i = 0; i < 5; ++i)
        if (chr == vowels[i])
            return true;

    return false;
}

string removeVowels(string str)
{
    string finalString = "";
    int length = str.length();
    for (int i = 0; i < length; ++i)
        if (!isVowel(tolower(str[i])))
            finalString += str[i];
    
    return finalString;
}

int main()
{
    string str = "";
    cout << "Provide some text: ";
    getline(cin, str);
    cout << "The text you entered is: " 
         << str << endl;

    cout << "Your text with vowels removed "
         << "is: " << removeVowels(str) << endl;
    return 0;
}
```
```
/* OUTPUT

Please provide some text:
inky pinky ponky
The text you entered is: inky pinky ponky
Your text with vowels removed is: nky pnky pnky
*/
```
### Finding the Roots of a Quadratic Equation
```
/*
Problem: Find the roots of a given quadratic equation.

A quadratic equation is an equation in the form ax^2 
+ bx + c = 0, where a is not equal to zero. The "roots"
of the quadratic are the numbers that satisfy the 
quadratic equation. There are always two roots for any
quadratic equation, although sometimes that may coincide.

The roots of any quadratic equation is given by: 
x = [-b +/- sqrt(-b^2 - 4ac)] / 2a

Steps:

1. Get the values of a, b, and c  
2. If a is equal to zero, the equation is not a quadratic equation  
3. Calculate the value of the discriminant: d = b^2 - (4ac)  
    1. If the discriminant is positive, then there are two distinct roots, both of which are real numbers.  
    2. If the discriminant is zero, then there is exactly one distinct real root, sometimes called a double root  
    3. If the discriminant is negative, then there are no real roots. Rather, there are two distinct (non-real) complex roots, which are complex conjugates of each other.  
*/
```
```cpp
#include <iostream>
#include <cmath>
using namespace std;

int main() 
{
    int a, b, c, d;
    double x1, x2;

    cout << "Program to find roots of quadratic equation" 
         << endl;
    cout << "Enter values for ax^2 + bx + c..." << endl;
    cout << "Enter value for a: ";
    cin >> a;
    cout << "Enter value for b: ";
    cin >> b;
    cout << "Enter value for c: ";
    cin >> c;
    cout << endl;

    if (a == 0)
        cout << "Not a quadratic equation.";

    else 
    {
        d = (b * b) - (4 * a * c);
        if (d > 0)
        {
            cout << "Real and distinct roots" << endl;
            x1 = ((-b + sqrt(d)) / (2 * a));
            x2 = ((-b - sqrt(d)) / (2 * a));

            cout << "Root 1 = " << x1 << endl;
            cout << "Root 2 = " << x2 << endl;
        }
        else if (d == 0)
        {
            cout << "Real and equal roots" << endl;
            x1 = x2 = -b / (2 * a);
            cout << "Root 1 = " << x1 << endl;
            cout << "Root 2 = " << x2 << endl;
        }
        else {
            cout << "Imaginary roots" << endl;
            x1 = -b / (2 * a);
            x2 = sqrt(-d) / (2 * a);
            cout << "Root 1 = " << x1 << endl;
            cout << "Root 2 = " << x2 << endl;
        }
    }
    return 0;
}
```
```
/* OUTPUT

Program to find roots of quadratic equation
Enter values for ax^2 + bx + c...
Enter value for a: 1
Enter value for b: -2
Enter value for c: -15

Real and distinct roots
Root 1 = 5
Root 2 = -3
*/
```
### Finding the Sum of Even-Valued Terms in the Fibonacci Sequence Less than 4 Million
```cpp
#include <iostream>
using namespace std;

int main() 
{
    cout << "By considering the terms in the Fibonacci "
         << "sequence whose values do not exceed four "
         << "million, find the sum of the even-valued "
         << "terms." << endl;

    int x = 0, y = 1, t = 0, sum = 0;
    while (y < 4000000)
    {
        if (y % 2 == 0) sum += y;

        t = x + y;
        x = y;
        y = t;
    }
    cout << "Answer: " << sum;
    return 0;
}
```
```
/* OUTPUT

Answer: 4613732
*/
```
### Finding the Sum of Even Element Values in a Matrix
```cpp
#include <iostream>
using namespace std;

int main()
{ 
    const int size = 4;
    int sum = 0;

    int matrix[size][size] = {{10, 11, 20, 22},
                              {30, 33, 40, 44},
                              {50, 55, 60, 66},
                              {70, 77, 80, 88}};

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            cout << matrix[i][j] << " ";
            if (matrix[i][j] % 2 == 0)
                sum += matrix[i][j];
        }
        cout << endl;
    }
    cout << "Sum of evens: " << sum << endl;
    return 0;
}
```
```
/* OUTPUT

10 11 20 22
30 33 40 44
50 55 60 66
70 77 80 88
Sum of evens: 580
*/
```
### The Tower of Hanoi
```
/* 
The Tower of Hanoi is a mathematical game or puzzle. It
consists of three rods, and a number of disks of different
sizes which can slide onto any rod. The puzzle starts with
the disks in a neat stack in ascending order of size on one
rod, the smallest at the top, thus making a conical shape.

The object of the puzzle is to move the entire stack to 
another rod, obeying the following rules:
1. Only one disk may be moved at a time  
2. Each move consists of taking the upper disk from one of the rods and sliding it onto another rod, on top of the other disks that may already be present on that rod.  
3. No disk may be placed on top of a smaller disk.  

A key to solving this puzzle is to recognize that it can be
solved by breaking the problem down into a collection of
smaller problems and further breaking those problems down 
into even smaller problems until a solution is reached.
The following procedure demonstrates this approach.
    * label the pegs A, B, C -- these labels may move at
      different steps
    * let n be the total number of discs
    * number the discs from 1 (smallest, topmost) to n
      (largest, bottommost)

To move n discs from peg A to peg C:
1. move n - 1 discs from A to B. This leaves disc n
       alone on peg A
2. move disc n from A to C
3. move n - 1 discs from B to C so they sit on disc n

The above is a recursive algorithm: to cary out steps 1 and
3, apply the same algorithm again for n - 1. The entire
procedure is a finite number of steps, since at some point
the algorithm will be required for n = 1. This step, moving 
a single disc from peg A to peg B, is trivial.
*/
```
```cpp
#include <iostream>
using namespace std;

void hanoi(int n);
void moveTower(int ht, char f, char t, char i);
void moveRing(int d, char f, char t);

int main()
{
    cout << "How many disks? ";
    int x; cin >> x;
    hanoi(x);
    return 0;
}

void hanoi(int n) 
{
    moveTower(n, 'A', 'B', 'C');
}

void moveTower(int ht, char f, char t, char i)
{
    if (ht > 0)
    {
        moveTower(ht - 1, f, i, t);
        moveRing(ht, f, t);
        moveTower(ht - 1, i, t, f);
    }
}

void moveRing(int d, char f, char t)
{
    cout << "Move ring " << d << " from ";
    cout << f << " to " << t << endl;
}
```
```
/* OUTPUT

How many disks? 3
Move ring 1 from A to B
Move ring 2 from A to C
Move ring 1 from B to C
Move ring 3 from A to B
Move ring 1 from C to A
Move ring 2 from C to B
Move ring 1 from A to B
*/
```
### Kruskal’s Algorithm
```
/*
Kruskal's algorithm is an algorithm in graph theory that
finds a minimum spanning tree for a connected weighted
graph. This means it finds a subset of the edges that forms
a tree that includes every vertex, where the total weight of
all the edges in the tree is minimized. If the graph is not 
connected, then it finds a minimum spanning forest (a
minimum spanning tree for each connected component).
Kruskal's algorithm is an example of a greedy algorithm.

Steps:

1. Create a forest F (a set of trees), where each vertex in the graph is a separate tree  
2. Create a set S containing all the edges in the graph  
3. While S is nonempty and F is not yet spanning  
    1. Remove an edge with minimum weight from S  
    2. If that edge connects two different trees, then add it to the forest, combining two trees into a single tree  
    3. Otherwise discard that edge  

At the termination of the algorithm, the forest has only one
component and forms a minimum spanning tree of the graph.
*/
```
```cpp
#include <iostream>
#include <algorithm>
using namespace std;

struct Edge 
{
    int m_first_vertex, m_second_vertex, m_weight;
};

bool checkCycle (Edge e, int path[]);

int main()
{
    // Create graph of 'n' vertices and 'm' edges
    cout << "Enter the number of vertices in the graph: ";
    int n = 0; cin >> n;    
    cout << "Enter the number of edges in the graph: ";
    int m = 0; cin >> m;

    int path[n + 1];
    struct Edge e[m];

    int i;
    cout << endl;
    for (i = 0; i < m; ++i)
    {
        cout << "Enter 2 vertices and weight of edge " 
             << i + 1 << endl;
        cout << "First vertex: ";
        cin >> e[i].m_first_vertex;
        cout << "Second vertex: ";
        cin >> e[i].m_second_vertex;
        cout << "Weight: ";
        cin >> e[i].m_weight;
        cout << endl;
    }

    // Sort the edges in ascending order of weights
    int j;
    for (i = 0; i <= (m - 1); ++i)
        for (j = 0; j < (m - i - 1); ++j)
            if (e[j].m_weight > e[j + 1].m_weight)
                swap (e[j], e[j + 1]);

    // Initialize the path array
    for (i = 1; i <= n; ++i)
        path[i] = 0;

    // Counts the number of edges selected in the tree
    i = 0;

    // Counts the number of edges selected or discarded
    j = 0;

    int minimum_cost = 0;

    while ((i != (n - 1)) && (j != m))
    {
        cout << "Edge (" << e[j].m_first_vertex << ", " 
             << e[j].m_second_vertex << ") with weight "
             << e[j].m_weight << " ";

        if (checkCycle (e[j], path))
        {
            minimum_cost += e[j].m_weight;
            i++;
            cout << "is selected";
        } 

        else cout << "is discarded";
        cout << endl;
        j++;
    }

    if (i != (n - 1))
        cout << "Minimum spanning tree cannot be formed";

    return 0;
}

bool checkCycle (Edge e, int path[])
{
    int first_vertex = e.m_first_vertex;
    int second_vertex = e.m_second_vertex;

    while (path[first_vertex] > 0) 
        first_vertex = path[first_vertex];

    while (path[second_vertex] > 0)
        second_vertex = path[second_vertex];

    if (first_vertex != second_vertex)
    {
        path[first_vertex] = second_vertex;
        return true;
    }
    return false;
}
```
```
/* OUTPUT

Enter the number of vertices in the graph: 4
Enter the number of edges in the graph: 4

Enter 2 vertices and weight of edge 1
First vertex: 1
Second vertex: 2
Weight: 6

Enter 2 vertices and weight of edge 2
First vertex: 1
Second vertex: 4
Weight: 5

Enter 2 vertices and weight of edge 3
First vertex: 1
Second vertex: 3
Weight: 7

Enter 2 vertices and weight of edge 4
First vertex: 3
Second vertex: 4
Weight 8

Edge (1, 4) with weight 5 is selected
Edge (1, 2) with weight 6 is selected
Edge (1, 3) with weight 7 is selected
*/
```
### Finding the Angle between the Hour Hand and the Minute Hand on an Analog Clock
```cpp
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

double angle(int hour, int minute)
{
    // Hour angle (from 12 o'clock): 360 * hours/12
    // ==> the hour hand moves at the rate of 30 degrees per hour
    //      or 0.5 degrees per minute
    double hour_angle = 0.5 * (hour * 60 + minute);

    // Minute angle (from 12 o'clock): 360 * minutes/60
    // ==> minute hand moves at the rate of 6 degrees per minute
    double minute_angle = 6 * minute;

    double angle_between = abs(hour_angle - minute_angle);
    angle_between = min(angle_between, 360 - angle_between);
    return angle_between;
}

int main() 
{
    cout << "Enter hour (1 - 12): ";
    int hour = 1; cin >> hour;
    cout << "Enter minute (0 - 59): ";
    int minute = 0; cin >> minute;
    cout << "Angle between hour hand and minute hand: "
         << angle(hour, minute) << " degrees" << endl;
    return 0;
}
```
```
/* VARIOUS OUTPUTS

Enter hour (1 - 12): 2
Enter minute (0 - 59): 20
Angle between hour hand and minute hand: 50 degrees

Enter hour (1 - 12): 3
Enter minute (0 - 59): 15
Angle between hour hand and minute hand: 7.5 degrees

Enter hour (1 - 12): 1
Enter minute (0 - 59): 0
Angle between hour hand and minute hand: 30 degrees

Enter hour (1 - 12): 6
Enter minute (0 - 59): 0
Angle between hour hand and minute hand: 180 degrees
*/
```
### What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
```
The following algorithm can be greatly optimized.
If n%20==0 then n%1==0, n%2==0, n%4==0, n%5==0, n%10==0.
If n%18==0 then n%3==0, n%6==0, n%9==0
If n%16==0 then n%8==0
If n%14==0 then n%7==0
The only values that are necessary to check are:
20, 19, 18, 17, 16, 15, 14, 13, 12, 11
An additional optimization is to divide from largest devisior to smallest as it is more likely for a number to be divisible by a smaller divisor so 20->11 is faster than 11->20.
```
```cpp
#include <iostream>
#include <ctime>
using namespace std;

int main() 
{
    int n, i, t;
    bool found = false;
    for (n = 1; !found; ++n)
    {
        found = true;
        //for (int i = 1; i <= 20; ++i)
        for (int i = 20; i >= 11; --i) // Optimization mentioned above
        {
            if (!(n%i)) 
            {
                found = false;
                break;
            }
        }
    }
    cout << n << endl;
    
    t = clock();
    cout << "The operation took " 
         << static_cast<double>(t)/CLOCKS_PER_SEC
         << " seconds." << endl;
    return 0;
}
```
```
/* OUTPUT

232792560
The operation took 3.28 seconds.
//It would run in about 1.7 seconds with the optimization.
*/
```
## Samples
### Hello World
```cpp
#include <iostream>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
cout << “Hello World” << endl;
return 0;
}
```
### Center Point of Arbitrary Points
```cpp
double centerX = sum(AllXPoints[])
double centerY = sum(AllYPoints[])
```

### Numeric - bool isdigit(char)
### Alphanumeric - bool isalnum(char)

## Input Handling

### Includes
```cpp
#include<sstream>
```
- String Stream
```cpp
#include<algorithm>
```
- remove_if
## Helpful Functions
### Character Checking
#### Check if a character is
- Alphabetic - `bool isalpha(char)`
- Numeric - `bool isdigit(char)`
- Alphanumeric - `bool isalnum(char)`

## Programming tricks
### Compile and Run (one line)
```
g++ myFile.cpp && ./a.out
```
### Use file for stdin
```
./a.out < file.in
```
### Send stdout to file
```
./a.out > file.out
```
### Simple complile and test script
```sh
rm -f $1;
g++ $1.cpp -o $1;
$1 < $1.in;
```
>Note the test input must be named `<executablename>.in`



## Misc.
### *nix - Time Command
>To use the command, simply precede any command by the word time, such as:
>>time ls  

>When the command completes, time will report how long it took to execute the ls command in terms of user CPU time, system CPU time, and real time. The output format varies between different versions of the command, and some give additional statistics  
>>$ time host wikipedia.org  
>>wikipedia.org has address 207.142.131.235  
>>0.000u 0.000s 0:00.17 0.0% 0+0k 0+0io 0pf+0w  
>>$  

>For more info, type “man time”  





### Map-of-maps iteration
```cpp
#include<map>
#include<utility>

std::pair <std::string,double> product1; // default constructor
std::pair <std::string,double> product2 ("tomatoes",2.30); // value init
std::pair <std::string,double> product3 (product2); // copy constructor


map<string, innerMap >::iterator it;
map<string,int >::iterator inner_it;
it=m.find(target);
for( inner_it=(*it).second.begin(); inner_it != (*it).second.end(); inner_it++)
{
    myQ.update_weight((*inner_it).first,(*it).first,(*inner_it).second);
}
```
## Graph Algorithms
### All-pairs shortest path algorithms
#### Floyd-Warshall O(n<sup>3</sup>)

##### Pseudocode
```
#setup
for i = 1 to N
   for j = 1 to N
      if there is an edge from i to j
         dist[0][i][j] = the length of the edge from i to j
      else
         dist[0][i][j] = INFINITY
#computation
for k = 1 to N
   for i = 1 to N
      for j = 1 to N
         dist[k][i][j] = min(dist[k-1][i][j], dist[k-1][i][k] + dist[k-1][k][j])

```

##### C++
```cpp

/**
 * Reads input from stdin into a matrix
 *
 * This expects the input to be formatted like so for a 3x3 matrix:
 * 1 2 3
 * 4 5 6
 * 7 8 9
 *
 * @param int n the size of our matrix (square matrix only for a graph)
 * @param vector< vector<int> > &m a matrix (2d vector)
 */
void read_input_to_matrix(int n, vector< vector<int> > &m) {
    int tmp;
    for (int i = 0; i < n; i++) {
        vector<int> ivec;
        for (int j = 0; j < n; j++) {
            cin >> tmp;
            ivec.push_back(tmp);
        }
        m.push_back(ivec);
    }
}

/**
 * Computes the all-pairs shortest path using Floyd-Warshall and updates our matrix
 * @param m the matrix of flight times
 */
void shortest_paths(vector< vector<int> > &m) {
    int n = m.size();
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (m[i][j] > m[i][k] + p[k] + m[k][j]) {
                    m[i][j] = m[i][k] + p[k] + m[k][j];
                }
            }
        }
    }
}
```
