/*
 * experiment_time_penalized_double_load.cpp
 *
 * Purpose:
 *   1. Generate random obstacle-grid delivery instances with target deadlines.
 *   2. Compare Distance-Reference, EDD-Greedy, Nearest-Greedy, Matching+EDD,
 *      Local Search, and Simulated Annealing (SA).
 *   3. Export manuscript-ready CSV files with English headers.
 *   4. Provide stronger reproducibility evidence through:
 *        - a matching-based scheduling baseline (Matching+EDD),
 *        - paired per-instance results for statistical testing,
 *        - small-scale exact-optimum comparisons by exhaustive permutation.
 *
 * Compile:
 *   g++ -std=c++17 -O2 experiment_time_penalized_double_load.cpp -o experiment
 *
 * Windows cmd:
 *   g++ -std=c++17 -O2 experiment_time_penalized_double_load.cpp -o experiment.exe
 *   experiment.exe
 *
 * Output files:
 *   table1_overall.csv
 *   table2_by_k.csv
 *   table3_by_beta.csv
 *   table4_by_lambda.csv
 *   table5_ablation.csv
 *   table6_extended_k.csv
 *   table7_extended_beta.csv
 *   table8_extended_lambda.csv
 *   table9_exact_gap.csv
 *   table10_paired_results.csv
 *
 * Note:
 *   For k <= exactMatchingLimit, Distance-Reference is computed by subset-DP
 *   minimum-distance matching and is a valid distance-only lower bound.
 *   For k > exactMatchingLimit, it falls back to a greedy distance reference,
 *   so it should not be interpreted as a strict lower bound.
 */

#include <bits/stdc++.h>
using namespace std;

using ll = long long;

static const int INF = 1e9;

struct Config {
    int n = 40;
    int m = 40;
    double obstacleRate = 0.18;

    vector<int> kValues = {8, 12, 16, 20, 24};
    vector<int> extendedKValues = {20, 24, 28, 32, 40, 48};
    vector<double> betaValues = {1.2, 1.6, 2.0, 2.4, 3.0};
    vector<double> extendedBetaValues = {2.0, 2.4, 3.0, 3.6};
    vector<double> lambdaValues = {0.5, 1.0, 2.0, 3.0, 5.0};
    vector<double> extendedLambdaValues = {2.0, 3.0, 5.0, 8.0};

    int instancesPerGroup = 30;
    int repeatsPerInstance = 5;
    int extendedInstancesPerGroup = 20;
    int exactMatchingLimit = 20;

    vector<int> exactKValues = {6, 8, 10};
    int exactInstancesPerGroup = 8;
    int exactRepeatsPerInstance = 3;
    int exactPermutationLimit = 10;

    int saIterations = 25000;
    int localIterations = 8000;

    double T0 = 200.0;
    double Tmin = 1e-4;
    double alpha = 0.9995;

    double pSwap = 0.45;
    double pBlockSwap = 0.35;
    double pReverse = 0.20;

    unsigned seed = 20260514;
};

struct Point {
    int x, y;
};

struct Instance {
    int n, m, k;
    vector<string> grid;
    Point S;
    vector<Point> X;
    vector<vector<int>> d; // d[0][i+1] = S to Xi, d[i+1][j+1] = Xi to Xj
    vector<int> deadline;
    double beta;
    double lambda;
};

struct Solution {
    double F = 0;
    int D = 0;
    int tardiness = 0;
    vector<int> perm;
};

struct Stats {
    double F = 0;
    double D = 0;
    double tardiness = 0;
    double lbGap = 0;
    double timeMs = 0;
    double stdF = 0;
    int count = 0;
};

struct RunValues {
    vector<double> FValues;
    vector<double> DValues;
    vector<double> tardValues;
    vector<double> gapValues;
    vector<double> timeValues;
};

double meanOf(const vector<double>& a) {
    if (a.empty()) return 0.0;
    double s = 0;
    for (double x : a) s += x;
    return s / (double)a.size();
}

double stdOf(const vector<double>& a) {
    if (a.size() <= 1) return 0.0;
    double mu = meanOf(a);
    double s = 0;
    for (double x : a) s += (x - mu) * (x - mu);
    return sqrt(s / (double)(a.size() - 1));
}

Stats summarize(const RunValues& rv) {
    Stats st;
    st.F = meanOf(rv.FValues);
    st.D = meanOf(rv.DValues);
    st.tardiness = meanOf(rv.tardValues);
    st.lbGap = meanOf(rv.gapValues);
    st.timeMs = meanOf(rv.timeValues);
    st.stdF = stdOf(rv.FValues);
    st.count = (int)rv.FValues.size();
    return st;
}

vector<vector<int>> bfsGrid(const vector<string>& grid, Point src) {
    int n = (int)grid.size();
    int m = (int)grid[0].size();
    vector<vector<int>> dist(n, vector<int>(m, INF));
    queue<Point> q;
    dist[src.x][src.y] = 0;
    q.push(src);

    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};

    while (!q.empty()) {
        auto cur = q.front();
        q.pop();

        for (int dir = 0; dir < 4; dir++) {
            int nx = cur.x + dx[dir];
            int ny = cur.y + dy[dir];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
            if (grid[nx][ny] == '#') continue;
            if (dist[nx][ny] != INF) continue;
            dist[nx][ny] = dist[cur.x][cur.y] + 1;
            q.push({nx, ny});
        }
    }
    return dist;
}

bool computeDistances(Instance& ins) {
    int k = ins.k;
    ins.d.assign(k + 1, vector<int>(k + 1, INF));

    vector<Point> pts;
    pts.push_back(ins.S);
    for (auto p : ins.X) pts.push_back(p);

    for (int i = 0; i <= k; i++) {
        auto dist = bfsGrid(ins.grid, pts[i]);
        for (int j = 0; j <= k; j++) {
            ins.d[i][j] = dist[pts[j].x][pts[j].y];
        }
    }

    for (int i = 1; i <= k; i++) {
        if (ins.d[0][i] >= INF) return false;
    }
    return true;
}
Instance generateInstance(
    int n,
    int m,
    int k,
    double obstacleRate,
    double beta,
    double lambda,
    mt19937& rng
) {
    uniform_real_distribution<double> ur(0.0, 1.0);
    uniform_int_distribution<int> rx(0, n - 1);
    uniform_int_distribution<int> ry(0, m - 1);

    while (true) {
        Instance ins;
        ins.n = n;
        ins.m = m;
        ins.k = k;
        ins.beta = beta;
        ins.lambda = lambda;
        ins.grid.assign(n, string(m, '.'));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (ur(rng) < obstacleRate) ins.grid[i][j] = '#';
            }
        }

        ins.S = {rx(rng), ry(rng)};
        ins.grid[ins.S.x][ins.S.y] = 'S';

        set<pair<int,int>> used;
        used.insert({ins.S.x, ins.S.y});

        int attempts = 0;
        while ((int)ins.X.size() < k && attempts < n * m * 20) {
            attempts++;
            int x = rx(rng);
            int y = ry(rng);
            if (ins.grid[x][y] == '#') continue;
            if (used.count({x, y})) continue;
            used.insert({x, y});
            ins.X.push_back({x, y});
            ins.grid[x][y] = 'T';
        }

        if ((int)ins.X.size() < k) continue;
        if (!computeDistances(ins)) continue;

        ins.deadline.assign(k, 0);
        for (int i = 0; i < k; i++) {
            int di = ins.d[0][i + 1];
            int low = (int)floor(-0.3 * di);
            int high = max(low + 1, (int)ceil(0.7 * di + 5));
            uniform_int_distribution<int> epsDist(low, high);
            int eps = epsDist(rng);
            int tau = max(1, (int)round(beta * di) + eps);
            ins.deadline[i] = tau;
        }

        return ins;
    }
}

Solution evaluatePerm(const Instance& ins, const vector<int>& perm) {
    Solution sol;
    sol.perm = perm;

    int t = 0;
    int D = 0;
    int tard = 0;
    int k = ins.k;

    for (int pos = 0; pos < k; ) {
        if (pos == k - 1) {
            int i = perm[pos];
            int di = ins.d[0][i + 1];

            int ci = t + di;
            int trip = 2 * di;

            D += trip;
            tard += max(0, ci - ins.deadline[i]);
            t += trip;
            pos += 1;
        } else {
            int i = perm[pos];
            int j = perm[pos + 1];

            int di = ins.d[0][i + 1];
            int dij = ins.d[i + 1][j + 1];
            int dj = ins.d[0][j + 1];

            int ci = t + di;
            int cj = t + di + dij;
            int trip = di + dij + dj;

            D += trip;
            tard += max(0, ci - ins.deadline[i]);
            tard += max(0, cj - ins.deadline[j]);
            t += trip;
            pos += 2;
        }
    }

    sol.D = D;
    sol.tardiness = tard;
    sol.F = (double)D + ins.lambda * (double)tard;
    return sol;
}


Solution exactOptimalByPermutation(const Instance& ins) {
    int k = ins.k;
    vector<int> p(k);
    iota(p.begin(), p.end(), 0);

    Solution best;
    best.F = numeric_limits<double>::infinity();

    do {
        Solution cur = evaluatePerm(ins, p);
        if (cur.F < best.F) best = cur;
    } while (next_permutation(p.begin(), p.end()));

    return best;
}

double improvementPct(double base, double improved) {
    if (fabs(base) < 1e-12) return 0.0;
    return (base - improved) / base * 100.0;
}

double gapPct(double value, double optimum) {
    if (fabs(optimum) < 1e-12) return 0.0;
    return (value - optimum) / optimum * 100.0;
}

vector<int> randomPerm(int k, mt19937& rng) {
    vector<int> p(k);
    iota(p.begin(), p.end(), 0);
    shuffle(p.begin(), p.end(), rng);
    return p;
}

vector<int> eddPerm(const Instance& ins) {
    vector<int> p(ins.k);
    iota(p.begin(), p.end(), 0);
    sort(p.begin(), p.end(), [&](int a, int b) {
        if (ins.deadline[a] != ins.deadline[b]) return ins.deadline[a] < ins.deadline[b];
        return ins.d[0][a + 1] < ins.d[0][b + 1];
    });
    return p;
}

vector<int> nearestGreedyPerm(const Instance& ins) {
    int k = ins.k;
    vector<int> rem(k);
    iota(rem.begin(), rem.end(), 0);
    vector<int> perm;

    int t = 0;
    while (!rem.empty()) {
        int first = -1;
        double bestScore = 1e100;

        for (int a : rem) {
            int arrival = t + ins.d[0][a + 1];
            double late = max(0, arrival - ins.deadline[a]);
            double score = ins.d[0][a + 1] + ins.lambda * late + 0.15 * ins.deadline[a];
            if (score < bestScore) {
                bestScore = score;
                first = a;
            }
        }

        perm.push_back(first);
        rem.erase(find(rem.begin(), rem.end(), first));

        if (rem.empty()) {
            t += 2 * ins.d[0][first + 1];
            break;
        }

        int second = -1;
        bestScore = 1e100;
        for (int b : rem) {
            int arrivalB = t + ins.d[0][first + 1] + ins.d[first + 1][b + 1];
            double late = max(0, arrivalB - ins.deadline[b]);
            double score = ins.d[first + 1][b + 1] + ins.d[0][b + 1]
                         + ins.lambda * late + 0.10 * ins.deadline[b];
            if (score < bestScore) {
                bestScore = score;
                second = b;
            }
        }

        perm.push_back(second);
        rem.erase(find(rem.begin(), rem.end(), second));

        t += ins.d[0][first + 1] + ins.d[first + 1][second + 1] + ins.d[0][second + 1];
    }

    return perm;
}

struct MatchingResult {
    int lb = INF;
    vector<pair<int,int>> pairs;
};

MatchingResult minDistanceMatchingDP(const Instance& ins) {
    int k = ins.k;
    int N = 1 << k;

    vector<int> dp(N, INF);
    vector<pair<int,int>> choice(N, {-1, -1});
    dp[0] = 0;

    for (int mask = 1; mask < N; mask++) {
        int i = -1;
        for (int x = 0; x < k; x++) {
            if (mask & (1 << x)) {
                i = x;
                break;
            }
        }

        int cnt = __builtin_popcount((unsigned)mask);
        if (cnt % 2 == 1) {
            int nmask = mask ^ (1 << i);
            int cost = 2 * ins.d[0][i + 1];
            if (dp[nmask] + cost < dp[mask]) {
                dp[mask] = dp[nmask] + cost;
                choice[mask] = {i, -1};
            }
        }

        for (int j = i + 1; j < k; j++) {
            if (!(mask & (1 << j))) continue;
            int nmask = mask ^ (1 << i) ^ (1 << j);
            int cost = ins.d[0][i + 1] + ins.d[i + 1][j + 1] + ins.d[0][j + 1];
            if (dp[nmask] + cost < dp[mask]) {
                dp[mask] = dp[nmask] + cost;
                choice[mask] = {i, j};
            }
        }
    }

    MatchingResult res;
    res.lb = dp[N - 1];

    int mask = N - 1;
    while (mask) {
        auto [a, b] = choice[mask];
        res.pairs.push_back({a, b});
        if (b == -1) mask ^= (1 << a);
        else mask ^= (1 << a) ^ (1 << b);
    }

    return res;
}

MatchingResult greedyDistanceMatching(const Instance& ins) {
    int k = ins.k;
    vector<int> rem(k);
    iota(rem.begin(), rem.end(), 0);

    MatchingResult res;
    res.lb = 0;

    while ((int)rem.size() >= 2) {
        int bestA = -1, bestB = -1;
        int bestCost = INF;

        for (int x = 0; x < (int)rem.size(); x++) {
            for (int y = x + 1; y < (int)rem.size(); y++) {
                int a = rem[x], b = rem[y];
                int cost = ins.d[0][a + 1] + ins.d[a + 1][b + 1] + ins.d[0][b + 1];
                if (cost < bestCost) {
                    bestCost = cost;
                    bestA = x;
                    bestB = y;
                }
            }
        }

        int a = rem[bestA];
        int b = rem[bestB];
        res.pairs.push_back({a, b});
        res.lb += bestCost;

        if (bestA > bestB) swap(bestA, bestB);
        rem.erase(rem.begin() + bestB);
        rem.erase(rem.begin() + bestA);
    }

    if (!rem.empty()) {
        int a = rem[0];
        res.pairs.push_back({a, -1});
        res.lb += 2 * ins.d[0][a + 1];
    }

    return res;
}

MatchingResult distanceReferenceMatching(const Instance& ins, int exactLimit) {
    if (ins.k <= exactLimit) {
        return minDistanceMatchingDP(ins);
    }
    return greedyDistanceMatching(ins);
}

vector<int> staticPairSortPerm(const Instance& ins) {
    auto mr = distanceReferenceMatching(ins, 20);
    vector<pair<int, pair<int,int>>> blocks;

    for (auto [a, b] : mr.pairs) {
        int key;
        if (b == -1) key = ins.deadline[a];
        else key = min(ins.deadline[a], ins.deadline[b]);
        blocks.push_back({key, {a, b}});
    }

    sort(blocks.begin(), blocks.end(), [](auto& u, auto& v) {
        return u.first < v.first;
    });

    vector<int> perm;
    for (auto& item : blocks) {
        int a = item.second.first;
        int b = item.second.second;
        if (b == -1) {
            perm.push_back(a);
        } else {
            if (ins.deadline[b] < ins.deadline[a]) swap(a, b);
            perm.push_back(a);
            perm.push_back(b);
        }
    }
    return perm;
}

void mutateSwap(vector<int>& p, mt19937& rng) {
    if (p.size() < 2) return;
    uniform_int_distribution<int> dist(0, (int)p.size() - 1);
    int a = dist(rng), b = dist(rng);
    while (b == a) b = dist(rng);
    swap(p[a], p[b]);
}

void mutateBlockSwap(vector<int>& p, mt19937& rng) {
    int k = (int)p.size();
    int blocks = (k + 1) / 2;
    if (blocks < 2) {
        mutateSwap(p, rng);
        return;
    }

    uniform_int_distribution<int> bd(0, blocks - 1);
    int A = bd(rng), B = bd(rng);
    while (B == A) B = bd(rng);

    int a1 = 2 * A;
    int b1 = 2 * B;

    int lenA = min(2, k - a1);
    int lenB = min(2, k - b1);

    if (lenA != lenB) {
        mutateSwap(p, rng);
        return;
    }

    for (int t = 0; t < lenA; t++) swap(p[a1 + t], p[b1 + t]);
}

void mutateReverse(vector<int>& p, mt19937& rng) {
    int k = (int)p.size();
    if (k < 2) return;
    uniform_int_distribution<int> dist(0, k - 1);
    int l = dist(rng), r = dist(rng);
    if (l > r) swap(l, r);
    if (l == r) return;
    reverse(p.begin() + l, p.begin() + r + 1);
}

Solution localSearch(const Instance& ins, vector<int> start, int iterations, mt19937& rng) {
    Solution cur = evaluatePerm(ins, start);
    Solution best = cur;

    uniform_real_distribution<double> ur(0.0, 1.0);

    for (int it = 0; it < iterations; it++) {
        vector<int> cand = cur.perm;
        double r = ur(rng);
        if (r < 0.55) mutateSwap(cand, rng);
        else if (r < 0.85) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);

        Solution nxt = evaluatePerm(ins, cand);
        if (nxt.F <= cur.F) {
            cur = nxt;
            if (cur.F < best.F) best = cur;
        }
    }
    return best;
}

struct SAOptions {
    bool useSwap = true;
    bool useBlockSwap = true;
    bool useReverse = true;
    double pSwap = 0.45;
    double pBlockSwap = 0.35;
    double pReverse = 0.20;
};

Solution simulatedAnnealing(
    const Instance& ins,
    vector<int> start,
    const Config& cfg,
    const SAOptions& opt,
    mt19937& rng
) {
    Solution cur = evaluatePerm(ins, start);
    Solution best = cur;

    double T = cfg.T0;
    uniform_real_distribution<double> ur(0.0, 1.0);

    for (int it = 0; it < cfg.saIterations && T > cfg.Tmin; it++) {
        vector<int> cand = cur.perm;

        vector<int> ops;
        if (opt.useSwap) ops.push_back(0);
        if (opt.useBlockSwap) ops.push_back(1);
        if (opt.useReverse) ops.push_back(2);
        if (ops.empty()) ops.push_back(0);

        double r = ur(rng);
        int op = 0;

        if (opt.useSwap && opt.useBlockSwap && opt.useReverse) {
            if (r < opt.pSwap) op = 0;
            else if (r < opt.pSwap + opt.pBlockSwap) op = 1;
            else op = 2;
        } else {
            uniform_int_distribution<int> od(0, (int)ops.size() - 1);
            op = ops[od(rng)];
        }

        if (op == 0) mutateSwap(cand, rng);
        else if (op == 1) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);

        Solution nxt = evaluatePerm(ins, cand);
        double delta = nxt.F - cur.F;

        if (delta <= 0 || ur(rng) < exp(-delta / max(1e-12, T))) {
            cur = nxt;
            if (cur.F < best.F) best = cur;
        }

        T *= cfg.alpha;
    }

    return best;
}

void addRun(RunValues& rv, const Solution& sol, int lb, double ms) {
    rv.FValues.push_back(sol.F);
    rv.DValues.push_back(sol.D);
    rv.tardValues.push_back(sol.tardiness);
    rv.gapValues.push_back((sol.F - lb) / max(1, lb) * 100.0);
    rv.timeValues.push_back(ms);
}

template <class Func>
Solution timedRun(Func f, double& ms) {
    auto st = chrono::steady_clock::now();
    Solution sol = f();
    auto ed = chrono::steady_clock::now();
    ms = chrono::duration<double, milli>(ed - st).count();
    return sol;
}

void writeOverallCSV(
    const string& filename,
    const map<string, Stats>& table
) {
    ofstream fout(filename);
    fout << "method,objective,distance,total_lateness,distance_ref_gap_pct,mean_time_ms,std_objective\n";
    for (auto& [name, st] : table) {
        fout << name << ","
             << fixed << setprecision(4)
             << st.F << ","
             << st.D << ","
             << st.tardiness << ","
             << st.lbGap << ","
             << st.timeMs << ","
             << st.stdF << "\n";
    }
}

void writeByKCSV(
    const string& filename,
    const vector<tuple<int, Stats, Stats, Stats, Stats>>& rows
) {
    ofstream fout(filename);
    fout << "k,edd_objective,matching_edd_objective,local_search_objective,sa_objective,sa_total_lateness,sa_mean_time_ms\n";
    for (auto& row : rows) {
        int k;
        Stats edd, statp, local, sa;
        tie(k, edd, statp, local, sa) = row;
        fout << k << ","
             << fixed << setprecision(4)
             << edd.F << ","
             << statp.F << ","
             << local.F << ","
             << sa.F << ","
             << sa.tardiness << ","
             << sa.timeMs << "\n";
    }
}

void writeByBetaCSV(
    const string& filename,
    const vector<tuple<double, Stats, Stats, Stats, Stats>>& rows
) {
    ofstream fout(filename);
    fout << "beta,edd_objective,matching_edd_objective,local_search_objective,sa_objective,sa_total_lateness\n";
    for (auto& row : rows) {
        double beta;
        Stats edd, statp, local, sa;
        tie(beta, edd, statp, local, sa) = row;
        fout << fixed << setprecision(4)
             << beta << ","
             << edd.F << ","
             << statp.F << ","
             << local.F << ","
             << sa.F << ","
             << sa.tardiness << "\n";
    }
}

void writeByLambdaCSV(
    const string& filename,
    const vector<tuple<double, Stats, Stats>>& rows
) {
    ofstream fout(filename);
    fout << "lambda,sa_distance,sa_total_lateness,sa_objective,matching_edd_objective\n";
    for (auto& row : rows) {
        double lambda;
        Stats sa, statp;
        tie(lambda, sa, statp) = row;
        fout << fixed << setprecision(4)
             << lambda << ","
             << sa.D << ","
             << sa.tardiness << ","
             << sa.F << ","
             << statp.F << "\n";
    }
}

void writeAblationCSV(
    const string& filename,
    const map<string, Stats>& table
) {
    ofstream fout(filename);
    fout << "method,swap,block_swap,segment_reverse,objective,total_lateness,mean_time_ms\n";
    auto yn = [](bool x) { return x ? "yes" : "no"; };

    struct Row {
        string name;
        bool sw, bl, rv;
    };

    vector<Row> rows = {
        {"SA-Swap", true, false, false},
        {"SA-SwapBlock", true, true, false},
        {"SA-Full", true, true, true}
    };

    for (auto& r : rows) {
        auto it = table.find(r.name);
        if (it == table.end()) continue;
        const Stats& st = it->second;
        fout << r.name << ","
             << yn(r.sw) << ","
             << yn(r.bl) << ","
             << yn(r.rv) << ","
             << fixed << setprecision(4)
             << st.F << ","
             << st.tardiness << ","
             << st.timeMs << "\n";
    }
}


struct ExactGapRow {
    int k = 0;
    Stats exactOpt;
    Stats matchingEDD;
    Stats local;
    Stats sa;
    double matchingGap = 0;
    double localGap = 0;
    double saGap = 0;
};

void writeExactGapCSV(const string& filename, const vector<ExactGapRow>& rows) {
    ofstream fout(filename);
    fout << "k,exact_optimum,matching_edd_objective,local_search_objective,sa_objective,"
         << "matching_edd_gap_pct,local_search_gap_pct,sa_gap_pct,exact_mean_time_ms,sa_mean_time_ms\n";
    for (const auto& r : rows) {
        fout << fixed << setprecision(4)
             << r.k << ","
             << r.exactOpt.F << ","
             << r.matchingEDD.F << ","
             << r.local.F << ","
             << r.sa.F << ","
             << r.matchingGap << ","
             << r.localGap << ","
             << r.saGap << ","
             << r.exactOpt.timeMs << ","
             << r.sa.timeMs << "\n";
    }
}

struct PairedRow {
    int instanceId = 0;
    int k = 0;
    double beta = 0;
    double lambda = 0;
    double eddObjective = 0;
    double nearestObjective = 0;
    double matchingEDDObjective = 0;
    double localObjective = 0;
    double saObjective = 0;
    double saVsLocalImprovement = 0;
    double saVsMatchingEDDImprovement = 0;
};

void writePairedResultsCSV(const string& filename, const vector<PairedRow>& rows) {
    ofstream fout(filename);
    fout << "instance_id,k,beta,lambda,edd_objective,nearest_greedy_objective,"
         << "matching_edd_objective,local_search_objective,sa_objective,"
         << "sa_vs_local_improvement_pct,sa_vs_matching_edd_improvement_pct\n";
    for (const auto& r : rows) {
        fout << fixed << setprecision(4)
             << r.instanceId << ","
             << r.k << ","
             << r.beta << ","
             << r.lambda << ","
             << r.eddObjective << ","
             << r.nearestObjective << ","
             << r.matchingEDDObjective << ","
             << r.localObjective << ","
             << r.saObjective << ","
             << r.saVsLocalImprovement << ","
             << r.saVsMatchingEDDImprovement << "\n";
    }
}

void printStatsLine(const string& name, const Stats& st) {
    cout << left << setw(18) << name
         << " F=" << setw(10) << fixed << setprecision(2) << st.F
         << " D=" << setw(8) << st.D
         << " Tard=" << setw(8) << st.tardiness
         << " Gap=" << setw(8) << st.lbGap
         << " Time(ms)=" << setw(8) << st.timeMs
         << " StdF=" << st.stdF << "\n";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Config cfg;
    mt19937 rng(cfg.seed);

    cout << "===== Time-Penalized Double-Load Grid Delivery Experiment =====\n";
    cout << "Grid: " << cfg.n << "x" << cfg.m
         << ", obstacleRate=" << cfg.obstacleRate << "\n";
    cout << "instancesPerGroup=" << cfg.instancesPerGroup
         << ", repeatsPerInstance=" << cfg.repeatsPerInstance << "\n";
    cout << "extendedInstancesPerGroup=" << cfg.extendedInstancesPerGroup
         << ", exactMatchingLimit=" << cfg.exactMatchingLimit << "\n";
    cout << "Note: for k > exactMatchingLimit, Distance-Ref is greedy reference, not exact LB.\n\n";

    {
        int k = 12;
        double beta = 1.6;
        double lambda = 1.0;

        map<string, RunValues> rv;
        RunValues lbRV;

        for (int instId = 0; instId < cfg.instancesPerGroup; instId++) {
            Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
            auto mr = distanceReferenceMatching(ins, 20);
            int lb = mr.lb;

            {
                RunValues& r = rv["Distance-LB"];
                r.FValues.push_back(0.0);
                r.DValues.push_back(lb);
                r.tardValues.push_back(0.0);
                r.gapValues.push_back(0.0);
                r.timeValues.push_back(0.0);
            }

            for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                double ms = 0;

                Solution edd = timedRun([&]() {
                    return evaluatePerm(ins, eddPerm(ins));
                }, ms);
                addRun(rv["EDD-Greedy"], edd, lb, ms);

                Solution near = timedRun([&]() {
                    return evaluatePerm(ins, nearestGreedyPerm(ins));
                }, ms);
                addRun(rv["Nearest-Greedy"], near, lb, ms);

                Solution statp = timedRun([&]() {
                    return evaluatePerm(ins, staticPairSortPerm(ins));
                }, ms);
                addRun(rv["Matching+EDD"], statp, lb, ms);

                Solution local = timedRun([&]() {
                    vector<int> start = eddPerm(ins);
                    return localSearch(ins, start, cfg.localIterations, rng);
                }, ms);
                addRun(rv["Local Search"], local, lb, ms);

                Solution sa = timedRun([&]() {
                    vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                    SAOptions opt;
                    opt.pSwap = cfg.pSwap;
                    opt.pBlockSwap = cfg.pBlockSwap;
                    opt.pReverse = cfg.pReverse;
                    return simulatedAnnealing(ins, start, cfg, opt, rng);
                }, ms);
                addRun(rv["SA"], sa, lb, ms);
            }
        }

        map<string, Stats> table;
        vector<string> order = {
            "Distance-LB", "EDD-Greedy", "Nearest-Greedy",
            "Matching+EDD", "Local Search", "SA"
        };

        cout << "Table 1 Overall:\n";
        for (auto& name : order) {
            table[name] = summarize(rv[name]);
            printStatsLine(name, table[name]);
        }
        cout << "\n";

        writeOverallCSV("table1_overall.csv", table);
    }

    // Table 10: paired per-instance results under the main setting.
    // These rows are designed for paired statistical tests and boxplots.
    {
        int k = 12;
        double beta = 1.6;
        double lambda = 1.0;
        vector<PairedRow> rows;

        for (int instId = 0; instId < cfg.instancesPerGroup; instId++) {
            Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
            int lb = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

            RunValues eddRV, nearestRV, matchingRV, localRV, saRV;
            for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                double ms = 0;

                Solution edd = timedRun([&]() { return evaluatePerm(ins, eddPerm(ins)); }, ms);
                addRun(eddRV, edd, lb, ms);

                Solution near = timedRun([&]() { return evaluatePerm(ins, nearestGreedyPerm(ins)); }, ms);
                addRun(nearestRV, near, lb, ms);

                Solution matching = timedRun([&]() { return evaluatePerm(ins, staticPairSortPerm(ins)); }, ms);
                addRun(matchingRV, matching, lb, ms);

                Solution local = timedRun([&]() { return localSearch(ins, eddPerm(ins), cfg.localIterations, rng); }, ms);
                addRun(localRV, local, lb, ms);

                Solution sa = timedRun([&]() {
                    vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                    SAOptions opt;
                    opt.pSwap = cfg.pSwap;
                    opt.pBlockSwap = cfg.pBlockSwap;
                    opt.pReverse = cfg.pReverse;
                    return simulatedAnnealing(ins, start, cfg, opt, rng);
                }, ms);
                addRun(saRV, sa, lb, ms);
            }

            Stats edd = summarize(eddRV);
            Stats nearest = summarize(nearestRV);
            Stats matching = summarize(matchingRV);
            Stats local = summarize(localRV);
            Stats sa = summarize(saRV);

            PairedRow row;
            row.instanceId = instId;
            row.k = k;
            row.beta = beta;
            row.lambda = lambda;
            row.eddObjective = edd.F;
            row.nearestObjective = nearest.F;
            row.matchingEDDObjective = matching.F;
            row.localObjective = local.F;
            row.saObjective = sa.F;
            row.saVsLocalImprovement = improvementPct(local.F, sa.F);
            row.saVsMatchingEDDImprovement = improvementPct(matching.F, sa.F);
            rows.push_back(row);
        }

        writePairedResultsCSV("table10_paired_results.csv", rows);
        cout << "Table 10 paired per-instance results exported.\n";
    }

    // Table 2: different k values
    {
        vector<tuple<int, Stats, Stats, Stats, Stats>> rows;

        for (int k : cfg.kValues) {
            double beta = 1.6;
            double lambda = 1.0;
            map<string, RunValues> rv;

            for (int instId = 0; instId < cfg.instancesPerGroup; instId++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int lb = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    double ms = 0;

                    Solution edd = timedRun([&]() {
                        return evaluatePerm(ins, eddPerm(ins));
                    }, ms);
                    addRun(rv["EDD-Greedy"], edd, lb, ms);

                    Solution statp = timedRun([&]() {
                        return evaluatePerm(ins, staticPairSortPerm(ins));
                    }, ms);
                    addRun(rv["Matching+EDD"], statp, lb, ms);

                    Solution local = timedRun([&]() {
                        return localSearch(ins, eddPerm(ins), cfg.localIterations, rng);
                    }, ms);
                    addRun(rv["Local Search"], local, lb, ms);

                    Solution sa = timedRun([&]() {
                        vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                        SAOptions opt;
                        return simulatedAnnealing(ins, start, cfg, opt, rng);
                    }, ms);
                    addRun(rv["SA"], sa, lb, ms);
                }
            }

            rows.push_back({
                k,
                summarize(rv["EDD-Greedy"]),
                summarize(rv["Matching+EDD"]),
                summarize(rv["Local Search"]),
                summarize(rv["SA"])
            });
        }

        writeByKCSV("table2_by_k.csv", rows);
        cout << "Table 2 by k exported.\n";
    }

    {
        vector<tuple<double, Stats, Stats, Stats, Stats>> rows;
        int k = 12;
        double lambda = 1.0;

        for (double beta : cfg.betaValues) {
            map<string, RunValues> rv;

            for (int instId = 0; instId < cfg.instancesPerGroup; instId++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int lb = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    double ms = 0;

                    Solution edd = timedRun([&]() {
                        return evaluatePerm(ins, eddPerm(ins));
                    }, ms);
                    addRun(rv["EDD-Greedy"], edd, lb, ms);

                    Solution statp = timedRun([&]() {
                        return evaluatePerm(ins, staticPairSortPerm(ins));
                    }, ms);
                    addRun(rv["Matching+EDD"], statp, lb, ms);

                    Solution local = timedRun([&]() {
                        return localSearch(ins, eddPerm(ins), cfg.localIterations, rng);
                    }, ms);
                    addRun(rv["Local Search"], local, lb, ms);

                    Solution sa = timedRun([&]() {
                        vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                        SAOptions opt;
                        return simulatedAnnealing(ins, start, cfg, opt, rng);
                    }, ms);
                    addRun(rv["SA"], sa, lb, ms);
                }
            }

            rows.push_back({
                beta,
                summarize(rv["EDD-Greedy"]),
                summarize(rv["Matching+EDD"]),
                summarize(rv["Local Search"]),
                summarize(rv["SA"])
            });
        }

        writeByBetaCSV("table3_by_beta.csv", rows);
        cout << "Table 3 by beta exported.\n";
    }

    {
        vector<tuple<double, Stats, Stats>> rows;
        int k = 12;
        double beta = 1.6;

        for (double lambda : cfg.lambdaValues) {
            map<string, RunValues> rv;

            for (int instId = 0; instId < cfg.instancesPerGroup; instId++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int lb = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    double ms = 0;

                    Solution statp = timedRun([&]() {
                        return evaluatePerm(ins, staticPairSortPerm(ins));
                    }, ms);
                    addRun(rv["Matching+EDD"], statp, lb, ms);

                    Solution sa = timedRun([&]() {
                        vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                        SAOptions opt;
                        return simulatedAnnealing(ins, start, cfg, opt, rng);
                    }, ms);
                    addRun(rv["SA"], sa, lb, ms);
                }
            }

            rows.push_back({
                lambda,
                summarize(rv["SA"]),
                summarize(rv["Matching+EDD"])
            });
        }

        writeByLambdaCSV("table4_by_lambda.csv", rows);
        cout << "Table 4 by lambda exported.\n";
    }

    {
        int k = 12;
        double beta = 1.6;
        double lambda = 1.0;

        map<string, RunValues> rv;

        for (int instId = 0; instId < cfg.instancesPerGroup; instId++) {
            Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
            int lb = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

            for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                double ms = 0;
                vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));

                SAOptions swapOnly;
                swapOnly.useSwap = true;
                swapOnly.useBlockSwap = false;
                swapOnly.useReverse = false;

                Solution s1 = timedRun([&]() {
                    return simulatedAnnealing(ins, start, cfg, swapOnly, rng);
                }, ms);
                addRun(rv["SA-Swap"], s1, lb, ms);

                SAOptions swapBlock;
                swapBlock.useSwap = true;
                swapBlock.useBlockSwap = true;
                swapBlock.useReverse = false;

                Solution s2 = timedRun([&]() {
                    return simulatedAnnealing(ins, start, cfg, swapBlock, rng);
                }, ms);
                addRun(rv["SA-SwapBlock"], s2, lb, ms);

                SAOptions full;
                full.useSwap = true;
                full.useBlockSwap = true;
                full.useReverse = true;

                Solution s3 = timedRun([&]() {
                    return simulatedAnnealing(ins, start, cfg, full, rng);
                }, ms);
                addRun(rv["SA-Full"], s3, lb, ms);
            }
        }

        map<string, Stats> table;
        table["SA-Swap"] = summarize(rv["SA-Swap"]);
        table["SA-SwapBlock"] = summarize(rv["SA-SwapBlock"]);
        table["SA-Full"] = summarize(rv["SA-Full"]);

        writeAblationCSV("table5_ablation.csv", table);
        cout << "Table 5 ablation exported.\n";
    }


    {
        vector<tuple<int, Stats, Stats, Stats, Stats>> rows;

        for (int k : cfg.extendedKValues) {
            double beta = 1.6;
            double lambda = 1.0;
            map<string, RunValues> rv;

            for (int instId = 0; instId < cfg.extendedInstancesPerGroup; instId++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    double ms = 0;

                    Solution edd = timedRun([&]() {
                        return evaluatePerm(ins, eddPerm(ins));
                    }, ms);
                    addRun(rv["EDD-Greedy"], edd, ref, ms);

                    Solution statp = timedRun([&]() {
                        return evaluatePerm(ins, staticPairSortPerm(ins));
                    }, ms);
                    addRun(rv["Matching+EDD"], statp, ref, ms);

                    Solution local = timedRun([&]() {
                        return localSearch(ins, eddPerm(ins), cfg.localIterations, rng);
                    }, ms);
                    addRun(rv["Local Search"], local, ref, ms);

                    Solution sa = timedRun([&]() {
                        vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                        SAOptions opt;
                        return simulatedAnnealing(ins, start, cfg, opt, rng);
                    }, ms);
                    addRun(rv["SA"], sa, ref, ms);
                }
            }

            rows.push_back({
                k,
                summarize(rv["EDD-Greedy"]),
                summarize(rv["Matching+EDD"]),
                summarize(rv["Local Search"]),
                summarize(rv["SA"])
            });
        }

        writeByKCSV("table6_extended_k.csv", rows);
        cout << "Table 6 extended k exported.\n";
    }

    {
        vector<tuple<double, Stats, Stats, Stats, Stats>> rows;
        int k = 12;
        double lambda = 1.0;

        for (double beta : cfg.extendedBetaValues) {
            map<string, RunValues> rv;

            for (int instId = 0; instId < cfg.extendedInstancesPerGroup; instId++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    double ms = 0;

                    Solution edd = timedRun([&]() {
                        return evaluatePerm(ins, eddPerm(ins));
                    }, ms);
                    addRun(rv["EDD-Greedy"], edd, ref, ms);

                    Solution statp = timedRun([&]() {
                        return evaluatePerm(ins, staticPairSortPerm(ins));
                    }, ms);
                    addRun(rv["Matching+EDD"], statp, ref, ms);

                    Solution local = timedRun([&]() {
                        return localSearch(ins, eddPerm(ins), cfg.localIterations, rng);
                    }, ms);
                    addRun(rv["Local Search"], local, ref, ms);

                    Solution sa = timedRun([&]() {
                        vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                        SAOptions opt;
                        return simulatedAnnealing(ins, start, cfg, opt, rng);
                    }, ms);
                    addRun(rv["SA"], sa, ref, ms);
                }
            }

            rows.push_back({
                beta,
                summarize(rv["EDD-Greedy"]),
                summarize(rv["Matching+EDD"]),
                summarize(rv["Local Search"]),
                summarize(rv["SA"])
            });
        }

        writeByBetaCSV("table7_extended_beta.csv", rows);
        cout << "Table 7 extended beta exported.\n";
    }

    {
        vector<tuple<double, Stats, Stats>> rows;
        int k = 12;
        double beta = 1.6;

        for (double lambda : cfg.extendedLambdaValues) {
            map<string, RunValues> rv;

            for (int instId = 0; instId < cfg.extendedInstancesPerGroup; instId++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    double ms = 0;

                    Solution statp = timedRun([&]() {
                        return evaluatePerm(ins, staticPairSortPerm(ins));
                    }, ms);
                    addRun(rv["Matching+EDD"], statp, ref, ms);

                    Solution sa = timedRun([&]() {
                        vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                        SAOptions opt;
                        return simulatedAnnealing(ins, start, cfg, opt, rng);
                    }, ms);
                    addRun(rv["SA"], sa, ref, ms);
                }
            }

            rows.push_back({
                lambda,
                summarize(rv["SA"]),
                summarize(rv["Matching+EDD"])
            });
        }

        writeByLambdaCSV("table8_extended_lambda.csv", rows);
        cout << "Table 8 extended lambda exported.\n";
    }


    // Table 9: small-scale exact-optimum comparison.
    // Exhaustive permutation is used only for small k to estimate optimality gaps.
    {
        vector<ExactGapRow> rows;
        double beta = 1.6;
        double lambda = 1.0;

        for (int k : cfg.exactKValues) {
            if (k > cfg.exactPermutationLimit) continue;

            RunValues exactRV, matchingRV, localRV, saRV;
            vector<double> matchingGaps, localGaps, saGaps;

            for (int instId = 0; instId < cfg.exactInstancesPerGroup; instId++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int lb = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;

                double exactMs = 0;
                Solution exact = timedRun([&]() { return exactOptimalByPermutation(ins); }, exactMs);
                addRun(exactRV, exact, lb, exactMs);

                double ms = 0;
                Solution matching = timedRun([&]() { return evaluatePerm(ins, staticPairSortPerm(ins)); }, ms);
                addRun(matchingRV, matching, lb, ms);
                matchingGaps.push_back(gapPct(matching.F, exact.F));

                Solution bestLocal;
                bestLocal.F = numeric_limits<double>::infinity();
                double localTimeSum = 0;
                for (int rep = 0; rep < cfg.exactRepeatsPerInstance; rep++) {
                    Solution cur = timedRun([&]() { return localSearch(ins, eddPerm(ins), cfg.localIterations, rng); }, ms);
                    localTimeSum += ms;
                    if (cur.F < bestLocal.F) bestLocal = cur;
                }
                addRun(localRV, bestLocal, lb, localTimeSum / cfg.exactRepeatsPerInstance);
                localGaps.push_back(gapPct(bestLocal.F, exact.F));

                Solution bestSA;
                bestSA.F = numeric_limits<double>::infinity();
                double saTimeSum = 0;
                for (int rep = 0; rep < cfg.exactRepeatsPerInstance; rep++) {
                    Solution cur = timedRun([&]() {
                        vector<int> start = (rep % 2 == 0 ? eddPerm(ins) : randomPerm(k, rng));
                        SAOptions opt;
                        opt.pSwap = cfg.pSwap;
                        opt.pBlockSwap = cfg.pBlockSwap;
                        opt.pReverse = cfg.pReverse;
                        return simulatedAnnealing(ins, start, cfg, opt, rng);
                    }, ms);
                    saTimeSum += ms;
                    if (cur.F < bestSA.F) bestSA = cur;
                }
                addRun(saRV, bestSA, lb, saTimeSum / cfg.exactRepeatsPerInstance);
                saGaps.push_back(gapPct(bestSA.F, exact.F));
            }

            ExactGapRow row;
            row.k = k;
            row.exactOpt = summarize(exactRV);
            row.matchingEDD = summarize(matchingRV);
            row.local = summarize(localRV);
            row.sa = summarize(saRV);
            row.matchingGap = meanOf(matchingGaps);
            row.localGap = meanOf(localGaps);
            row.saGap = meanOf(saGaps);
            rows.push_back(row);
        }

        writeExactGapCSV("table9_exact_gap.csv", rows);
        cout << "Table 9 exact-optimum gap results exported.\n";
    }

    cout << "\nAll experiments finished. CSV files generated in the current directory.\n";
    return 0;
}
