/*
 * ==========================================================================
 *  Traffic-Aware Vehicle Routing  (beginner-friendly, corrected version)
 * ==========================================================================
 *
 *  Idea in one line:
 *      Send cars ONE BY ONE (requests take turns). Every car already on a
 *      road makes that road look ALPHA units longer to the next car.
 *      Then run normal Dijkstra.
 *
 *  Three algorithms are compared on the SAME networks and requests:
 *
 *   1. Shortest Path (SP)   : every car takes the shortest path.
 *                             road cost = distance
 *
 *   2. Greedy Load-Balance  : every car takes the path through the
 *                             emptiest roads (distance is ignored).
 *                             road cost = load + 1
 *
 *   3. Proposed (Hybrid)    : mix of both.
 *                             road cost = distance + ALPHA * load
 *                             ALPHA = 0   -> exactly Shortest Path
 *                             ALPHA big   -> spreads traffic like Greedy
 *
 *  Why "distance + ALPHA * load" and not "distance * (1 + ALPHA * load)"?
 *      All roads are the same type, so 20 cars on a 1 km road is just as
 *      bad a jam as 20 cars on a 20 km road. With the multiplied version a
 *      short road hardly gets more expensive, so it stays jammed however
 *      big ALPHA is. With the added version every road gets the same
 *      penalty per car.
 *
 *  Compile:  g++ -O2 -std=c++17 traffic_routing.cpp -o traffic
 *  Run:      ./traffic
 * ==========================================================================
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <set>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
using namespace std;

// --------------------------------------------------------------------------
// PART 1 : Basic data structures
// --------------------------------------------------------------------------

struct Edge {            // one road
    int u, v;            // its two end intersections
    int w;               // its length (distance)
};

struct Graph {
    int V;                                   // number of intersections
    vector<Edge> edges;                      // all roads, edge id = index
    vector<vector<pair<int,int>>> adj;       // adj[u] = list of (neighbour, edgeId)
};

struct Request {         // r = (n, s, d)
    int n;               // number of vehicles
    int s;               // source
    int d;               // destination
};

struct PathShare {       // "x vehicles of this request go on this path"
    vector<int> edgeIds;
    int vehicles;
};

struct RequestResult {
    Request req;
    vector<PathShare> paths;
};

struct StepResult {      // result of routing ONE time step
    vector<RequestResult> results;
    vector<int> load;    // load[e] = number of vehicles using road e
    long long totalDistance;
    int unrouted;        // cars with no route (only possible on a disconnected map)
};

struct Metrics {
    int    maxLoad;
    double avgLoad;
    double variance;
    long long totalDistance;
};

const int SHORTEST = 0, GREEDY = 1, PROPOSED = 2;
const string ALGO_NAME[3] = {"Shortest Path", "Greedy Load-Bal", "Proposed Hybrid"};
const double DEFAULT_ALPHA = 3.0;     // extra length per car already on a road
                                      // (road lengths are 1..20)

// --------------------------------------------------------------------------
// PART 2 : Generate a random CONNECTED road network and random requests
// --------------------------------------------------------------------------

int randomInt(int lo, int hi, mt19937& rng) {      // random number in [lo, hi]
    return uniform_int_distribution<int>(lo, hi)(rng);
}

void addRoad(Graph& g, int u, int v, int w) {
    int id = g.edges.size();
    g.edges.push_back({u, v, w});
    g.adj[u].push_back({v, id});   // undirected: add both directions
    g.adj[v].push_back({u, id});
}

// Step 1: put all intersections on one big loop, in random order.
//         -> the network is connected, AND every road has a detour
//            (no dead-end road that all cars are forced to use).
// Step 2: add some extra random roads (more alternative routes).
Graph generateGraph(int V, int extraEdges, mt19937& rng) {
    Graph g;
    g.V = V;
    g.adj.assign(V, {});
    set<pair<int,int>> used;                       // avoid duplicate roads

    vector<int> order(V);
    for (int i = 0; i < V; i++) order[i] = i;
    shuffle(order.begin(), order.end(), rng);

    for (int i = 0; i < V; i++) {                  // loop part
        int u = order[i], v = order[(i + 1) % V];
        if (u == v || used.count({min(u, v), max(u, v)})) continue;  // tiny V
        addRoad(g, u, v, randomInt(1, 20, rng));
        used.insert({min(u, v), max(u, v)});
    }

    int added = 0, tries = 0;
    while (added < extraEdges && tries < extraEdges * 20) {  // extra roads
        tries++;
        int u = randomInt(0, V - 1, rng), v = randomInt(0, V - 1, rng);
        if (u == v) continue;
        if (used.count({min(u, v), max(u, v)})) continue;
        addRoad(g, u, v, randomInt(1, 20, rng));
        used.insert({min(u, v), max(u, v)});
        added++;
    }
    return g;
}

vector<Request> generateRequests(int V, int count, int minVeh, int maxVeh, mt19937& rng) {
    vector<Request> reqs;
    if (V < 2) return reqs;                        // need two different places
    for (int i = 0; i < count; i++) {
        int s = randomInt(0, V - 1, rng), d = randomInt(0, V - 1, rng);
        while (d == s) d = randomInt(0, V - 1, rng);
        reqs.push_back({randomInt(minVeh, maxVeh, rng), s, d});
    }
    return reqs;
}

// --------------------------------------------------------------------------
// PART 3 : Dijkstra with a custom cost for every road
//          Puts the path (list of edge ids, from s to d) into 'path'.
//          Returns false if d cannot be reached from s.
// --------------------------------------------------------------------------

bool dijkstra(const Graph& g, int s, int d, const vector<double>& cost, vector<int>& path) {
    const double INF = 1e18;
    vector<double> dist(g.V, INF);
    vector<int> parentEdge(g.V, -1);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<pair<double,int>>> pq;

    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        double du = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        if (du > dist[u]) continue;       // old entry, skip
        if (u == d) break;                // reached destination, stop early
        for (auto& nb : g.adj[u]) {
            int v = nb.first, id = nb.second;
            if (dist[u] + cost[id] < dist[v]) {
                dist[v] = dist[u] + cost[id];
                parentEdge[v] = id;
                pq.push({dist[v], v});
            }
        }
    }

    path.clear();
    if (dist[d] == INF) return false;     // d is not reachable

    // walk back from d to s using parentEdge
    int cur = d;
    while (cur != s) {
        int id = parentEdge[cur];
        path.push_back(id);
        cur = (g.edges[id].u == cur) ? g.edges[id].v : g.edges[id].u;
    }
    reverse(path.begin(), path.end());
    return true;
}

// --------------------------------------------------------------------------
// PART 4 : The routing algorithms (all three share this one function)
// --------------------------------------------------------------------------

// What one road "costs" for the next car, given how many cars already use it.
double roadCost(const Edge& e, int load, int mode, double alpha) {
    if (mode == SHORTEST) return e.w;
    if (mode == GREEDY)   return load + 1;
    return e.w + alpha * load;                       // PROPOSED
}

// Add 'cars' vehicles on 'path' to the request's list (same paths grouped).
void addToRequest(RequestResult& rr, const vector<int>& path, int cars) {
    for (auto& ps : rr.paths)
        if (ps.edgeIds == path) { ps.vehicles += cars; return; }
    rr.paths.push_back({path, cars});
}

StepResult routeOneStep(const Graph& g, const vector<Request>& reqs, int mode, double alpha) {
    int E = g.edges.size();
    int R = reqs.size();
    StepResult res;
    res.load.assign(E, 0);              // every time step starts with empty roads
    res.totalDistance = 0;
    res.unrouted = 0;
    res.results.resize(R);
    for (int i = 0; i < R; i++) res.results[i].req = reqs[i];

    vector<double> cost(E);             // cost of every road for the next car
    for (int e = 0; e < E; e++) cost[e] = roadCost(g.edges[e], 0, mode, alpha);

    vector<int> carsLeft(R);
    for (int i = 0; i < R; i++) carsLeft[i] = reqs[i].n;

    // Requests take turns: one car of request 0, one of request 1, ...,
    // then again from request 0. (Sending ALL cars of request 0 first would
    // let it grab the empty roads before the other requests get a chance.)
    bool carsRemain = true;
    while (carsRemain) {
        carsRemain = false;
        for (int i = 0; i < R; i++) {
            if (carsLeft[i] == 0) continue;
            carsRemain = true;

            // Shortest Path ignores traffic, so every car of a request gets
            // the same path: route them all with ONE Dijkstra.
            int cars = (mode == SHORTEST) ? carsLeft[i] : 1;
            carsLeft[i] -= cars;

            // 1) find the cheapest path with the current road costs
            vector<int> path;
            if (!dijkstra(g, reqs[i].s, reqs[i].d, cost, path)) {
                res.unrouted += cars;
                continue;
            }

            // 2) remember it (group identical paths together)
            addToRequest(res.results[i], path, cars);

            // 3) put the car(s) on the roads -> loads and costs go up
            for (int id : path) {
                res.load[id] += cars;
                res.totalDistance += (long long)cars * g.edges[id].w;
                cost[id] = roadCost(g.edges[id], res.load[id], mode, alpha);
            }
        }
    }
    return res;
}

// --------------------------------------------------------------------------
// PART 5 : Evaluation metrics
// --------------------------------------------------------------------------

Metrics computeMetrics(const StepResult& res) {
    Metrics m;
    int E = res.load.size();
    m.maxLoad = 0;
    m.avgLoad = 0;
    m.variance = 0;
    m.totalDistance = res.totalDistance;
    if (E == 0) return m;
    double sum = 0;
    for (int x : res.load) { m.maxLoad = max(m.maxLoad, x); sum += x; }
    m.avgLoad = sum / E;
    double sq = 0;
    for (int x : res.load) sq += (x - m.avgLoad) * (x - m.avgLoad);
    m.variance = sq / E;
    return m;
}

// --------------------------------------------------------------------------
// PART 6 : Printing helpers
// --------------------------------------------------------------------------

string pathToString(const Graph& g, int s, const vector<int>& path) {
    string out = to_string(s);
    int cur = s, len = 0;
    for (int id : path) {
        cur = (g.edges[id].u == cur) ? g.edges[id].v : g.edges[id].u;
        out += " -> " + to_string(cur);
        len += g.edges[id].w;
    }
    return out + "   (length " + to_string(len) + ")";
}

void printLine(char c = '-', int n = 78) { cout << string(n, c) << "\n"; }

void printMetricsRow(const string& name, const Metrics& m, double timeMs = -1) {
    cout << left << setw(18) << name << right
         << setw(8)  << m.maxLoad
         << setw(10) << fixed << setprecision(2) << m.avgLoad
         << setw(11) << m.variance
         << setw(12) << m.totalDistance;
    if (timeMs >= 0) cout << setw(12) << setprecision(2) << timeMs;
    cout << "\n";
}

void printMetricsHeader(bool withTime) {
    cout << left << setw(18) << "Algorithm" << right
         << setw(8) << "MaxLoad" << setw(10) << "AvgLoad"
         << setw(11) << "Variance" << setw(12) << "TotalDist";
    if (withTime) cout << setw(12) << "Time(ms)";
    cout << "\n";
}

// --------------------------------------------------------------------------
// PART 7 : Experiments
// --------------------------------------------------------------------------

// (A) Small network: print EVERYTHING (paths, distribution, road loads)
void detailedDemo(mt19937& rng) {
    printLine('=');
    cout << "PART A : DETAILED REPORT ON A SMALL NETWORK  (ALPHA = " << DEFAULT_ALPHA << ")\n";
    printLine('=');

    Graph g = generateGraph(8, 6, rng);
    cout << "Network: " << g.V << " intersections, " << g.edges.size() << " roads\n";
    for (int i = 0; i < (int)g.edges.size(); i++)
        cout << "  Road " << setw(2) << i << ": " << g.edges[i].u << " -- "
             << g.edges[i].v << "  (distance " << g.edges[i].w << ")\n";

    int timeSteps = 3;
    for (int t = 1; t <= timeSteps; t++) {
        vector<Request> reqs = generateRequests(g.V, 4, 5, 15, rng);
        cout << "\n";
        printLine('#');
        cout << "TIME STEP " << t << "   requests:";
        for (auto& r : reqs) cout << "  (n=" << r.n << ", " << r.s << "->" << r.d << ")";
        cout << "\n";
        printLine('#');

        StepResult res[3];
        for (int a = 0; a < 3; a++) {
            res[a] = routeOneStep(g, reqs, a, DEFAULT_ALPHA);
            cout << "\n[" << ALGO_NAME[a] << "]  assigned paths:\n";
            for (auto& rr : res[a].results) {
                cout << "  Request (" << rr.req.n << " vehicles, " << rr.req.s
                     << " -> " << rr.req.d << ")\n";
                for (auto& ps : rr.paths)
                    cout << "     " << setw(3) << ps.vehicles << " vehicles : "
                         << pathToString(g, rr.req.s, ps.edgeIds) << "\n";
            }
            if (res[a].unrouted > 0)
                cout << "  WARNING: " << res[a].unrouted << " vehicles had no route\n";
        }

        cout << "\nRoad-wise traffic load (vehicles on each road):\n";
        cout << "  Road  (u--v)    SP  Greedy  Proposed\n";
        for (int e = 0; e < (int)g.edges.size(); e++)
            cout << "  " << setw(4) << e << "  (" << g.edges[e].u << "--" << g.edges[e].v << ")"
                 << setw(8) << res[0].load[e] << setw(8) << res[1].load[e]
                 << setw(10) << res[2].load[e] << "\n";

        cout << "\nMetrics for time step " << t << ":\n";
        printMetricsHeader(false);
        for (int a = 0; a < 3; a++) printMetricsRow(ALGO_NAME[a], computeMetrics(res[a]));
    }
}

// (B) Bigger networks: only summary numbers + running time
void scalabilityTest(mt19937& rng) {
    cout << "\n";
    printLine('=');
    cout << "PART B : COMPARISON ON LARGER NETWORKS (5 time steps each)\n";
    cout << "  MaxLoad  = largest road load in a time step, averaged over the 5 steps\n";
    cout << "  AvgLoad, Variance = averaged over the 5 steps\n";
    cout << "  TotalDist = sum over all 5 steps, Time = total running time\n";
    printLine('=');

    int sizes[] = {50, 200, 1000, 5000};
    int timeSteps = 5, requestsPerStep = 30;

    for (int V : sizes) {
        Graph g = generateGraph(V, V, rng);      // about 2V roads
        vector<vector<Request>> allReqs;          // same requests for all algorithms
        for (int t = 0; t < timeSteps; t++)
            allReqs.push_back(generateRequests(V, requestsPerStep, 5, 20, rng));

        cout << "\nNetwork: " << V << " nodes, " << g.edges.size() << " roads\n";
        printMetricsHeader(true);

        for (int a = 0; a < 3; a++) {
            double sumMax = 0, sumAvg = 0, sumVar = 0;
            long long dist = 0;
            auto start = chrono::high_resolution_clock::now();
            for (int t = 0; t < timeSteps; t++) {
                Metrics m = computeMetrics(routeOneStep(g, allReqs[t], a, DEFAULT_ALPHA));
                sumMax += m.maxLoad; sumAvg += m.avgLoad; sumVar += m.variance;
                dist += m.totalDistance;
            }
            auto end = chrono::high_resolution_clock::now();
            double ms = chrono::duration<double, milli>(end - start).count();

            Metrics avg;
            avg.maxLoad = (int)(sumMax / timeSteps + 0.5);
            avg.avgLoad = sumAvg / timeSteps;
            avg.variance = sumVar / timeSteps;
            avg.totalDistance = dist;
            printMetricsRow(ALGO_NAME[a], avg, ms);
        }
    }
}

// (C) Trade-off: how ALPHA changes balance vs distance
void tradeOffTest(mt19937& rng) {
    cout << "\n";
    printLine('=');
    cout << "PART C : TRADE-OFF (200 nodes, 5 time steps; MaxLoad & Variance averaged,\n";
    cout << "         TotalDist summed over the 5 steps)\n";
    printLine('=');

    int timeSteps = 5;
    Graph g = generateGraph(200, 200, rng);
    vector<vector<Request>> allReqs;
    for (int t = 0; t < timeSteps; t++)
        allReqs.push_back(generateRequests(200, 40, 10, 30, rng));

    // runs one algorithm on all time steps and prints one line
    auto runAndPrint = [&](const string& label, int mode, double alpha) {
        double sumMax = 0, sumVar = 0;
        long long dist = 0;
        for (int t = 0; t < timeSteps; t++) {
            Metrics m = computeMetrics(routeOneStep(g, allReqs[t], mode, alpha));
            sumMax += m.maxLoad; sumVar += m.variance; dist += m.totalDistance;
        }
        cout << left << setw(18) << label << right << fixed << setprecision(1)
             << setw(8) << sumMax / timeSteps << setw(11) << setprecision(2) << sumVar / timeSteps
             << setw(12) << dist << "\n";
    };

    cout << left << setw(18) << "Algorithm" << right << setw(8) << "MaxLoad"
         << setw(11) << "Variance" << setw(12) << "TotalDist" << "\n";
    runAndPrint("Shortest Path", SHORTEST, 0);
    double alphas[] = {0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0};
    for (double alpha : alphas) {
        ostringstream label;
        label << "Hybrid ALPHA=" << alpha;
        runAndPrint(label.str(), PROPOSED, alpha);
    }
    runAndPrint("Greedy Load-Bal", GREEDY, 0);
    cout << "(ALPHA = 0 is exactly Shortest Path. Bigger ALPHA -> lower max load and\n"
         << " variance, but cars take longer detours, so total distance grows.)\n";
}

int main() {
    mt19937 rng(42);          // fixed seed -> same output every run
    detailedDemo(rng);
    scalabilityTest(rng);
    tradeOffTest(rng);
    return 0;
}
