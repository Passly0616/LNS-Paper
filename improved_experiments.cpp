#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

static const int INF = 1000000000;

struct Config {
    int n = 40;
    int m = 40;
    double obstacleRate = 0.18;
    vector<int> kValues = {16, 24, 32, 40, 48, 64};
    int instancesPerK = 20;
    int repeatsPerInstance = 3;
    int budgetMs = 300;
    unsigned baseSeed = 20260524;
    string outDir = "results";
};

struct Point { int x = 0, y = 0; };

struct Instance {
    int n = 0, m = 0, k = 0;
    vector<string> grid;
    Point depot;
    vector<Point> targets;
    vector<vector<int>> d;
    vector<int> deadline;
    double beta = 1.6;
    double lambda = 1.0;
};

struct Solution {
    double F = 0;
    int D = 0;
    int tardiness = 0;
    vector<int> perm;
};

struct ResultRow {
    string experiment;
    int k = 0;
    int instanceId = 0;
    int runId = 0;
    unsigned instanceSeed = 0;
    unsigned runSeed = 0;
    string algorithm;
    string variant;
    double budgetMs = 0;
    double timeMs = 0;
    double F = 0;
    int D = 0;
    int tardiness = 0;
    double gapToBestPct = 0;
};

struct SummaryRow {
    string experiment;
    int k = 0;
    string algorithm;
    string variant;
    int count = 0;
    double budgetMs = 0;
    double meanF = 0;
    double stdF = 0;
    double meanD = 0;
    double meanTardiness = 0;
    double meanTimeMs = 0;
    double meanGapToBestPct = 0;
    double winRatePct = 0;
};

struct LnsOptions {
    bool useRegret = true;
    bool adaptiveWeights = true;
    bool localImprove = true;
    bool acceptWorse = true;
    double destroyRatio = 0.30;
};

static double mean(const vector<double>& v) {
    if (v.empty()) return 0.0;
    return accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

static double stdev(const vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mu = mean(v), s = 0.0;
    for (double x : v) s += (x - mu) * (x - mu);
    return sqrt(s / (double)(v.size() - 1));
}

static string csvEscape(const string& s) {
    if (s.find_first_of(",\"\n\r") == string::npos) return s;
    string out = "\"";
    for (char c : s) out += (c == '"' ? "\"\"" : string(1, c));
    out += "\"";
    return out;
}

static string latexEscape(const string& s) {
    string out;
    for (char c : s) {
        if (c == '_') out += "\\_";
        else if (c == '%') out += "\\%";
        else if (c == '&') out += "\\&";
        else out += c;
    }
    return out;
}

static vector<vector<int>> bfsGrid(const vector<string>& grid, Point src) {
    int n = (int)grid.size(), m = (int)grid[0].size();
    vector<vector<int>> dist(n, vector<int>(m, INF));
    queue<Point> q;
    dist[src.x][src.y] = 0;
    q.push(src);
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        Point cur = q.front();
        q.pop();
        for (int dir = 0; dir < 4; dir++) {
            int nx = cur.x + dx[dir], ny = cur.y + dy[dir];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
            if (grid[nx][ny] == '#') continue;
            if (dist[nx][ny] != INF) continue;
            dist[nx][ny] = dist[cur.x][cur.y] + 1;
            q.push({nx, ny});
        }
    }
    return dist;
}

static bool computeDistances(Instance& ins) {
    vector<Point> pts;
    pts.push_back(ins.depot);
    for (Point p : ins.targets) pts.push_back(p);
    int k = ins.k;
    ins.d.assign(k + 1, vector<int>(k + 1, INF));
    for (int i = 0; i <= k; i++) {
        auto dist = bfsGrid(ins.grid, pts[i]);
        for (int j = 0; j <= k; j++) ins.d[i][j] = dist[pts[j].x][pts[j].y];
    }
    for (int i = 1; i <= k; i++) if (ins.d[0][i] >= INF) return false;
    return true;
}

static Instance generateInstance(const Config& cfg, int k, unsigned seed) {
    mt19937 rng(seed);
    uniform_real_distribution<double> ur(0.0, 1.0);
    uniform_int_distribution<int> rx(0, cfg.n - 1), ry(0, cfg.m - 1);
    while (true) {
        Instance ins;
        ins.n = cfg.n;
        ins.m = cfg.m;
        ins.k = k;
        ins.beta = 1.6;
        ins.lambda = 1.0;
        ins.grid.assign(cfg.n, string(cfg.m, '.'));
        for (int i = 0; i < cfg.n; i++) {
            for (int j = 0; j < cfg.m; j++) {
                if (ur(rng) < cfg.obstacleRate) ins.grid[i][j] = '#';
            }
        }
        ins.depot = {rx(rng), ry(rng)};
        ins.grid[ins.depot.x][ins.depot.y] = 'S';
        set<pair<int, int>> used;
        used.insert({ins.depot.x, ins.depot.y});
        int attempts = 0;
        while ((int)ins.targets.size() < k && attempts < cfg.n * cfg.m * 40) {
            attempts++;
            int x = rx(rng), y = ry(rng);
            if (ins.grid[x][y] == '#') continue;
            if (used.count({x, y})) continue;
            used.insert({x, y});
            ins.targets.push_back({x, y});
            ins.grid[x][y] = 'T';
        }
        if ((int)ins.targets.size() < k) continue;
        if (!computeDistances(ins)) continue;
        ins.deadline.assign(k, 0);
        for (int i = 0; i < k; i++) {
            int di = ins.d[0][i + 1];
            int low = (int)floor(-0.3 * di);
            int high = max(low + 1, (int)ceil(0.7 * di + 5));
            uniform_int_distribution<int> eps(low, high);
            ins.deadline[i] = max(1, (int)round(ins.beta * di) + eps(rng));
        }
        return ins;
    }
}

static Solution evaluatePerm(const Instance& ins, const vector<int>& perm) {
    Solution sol;
    sol.perm = perm;
    int t = 0, D = 0, tard = 0;
    for (int pos = 0; pos < (int)perm.size();) {
        if (pos == (int)perm.size() - 1) {
            int i = perm[pos], di = ins.d[0][i + 1];
            D += 2 * di;
            tard += max(0, t + di - ins.deadline[i]);
            t += 2 * di;
            pos++;
        } else {
            int i = perm[pos], j = perm[pos + 1];
            int di = ins.d[0][i + 1], dij = ins.d[i + 1][j + 1], dj = ins.d[0][j + 1];
            D += di + dij + dj;
            tard += max(0, t + di - ins.deadline[i]);
            tard += max(0, t + di + dij - ins.deadline[j]);
            t += di + dij + dj;
            pos += 2;
        }
    }
    sol.D = D;
    sol.tardiness = tard;
    sol.F = (double)D + ins.lambda * tard;
    return sol;
}

static vector<int> eddPerm(const Instance& ins) {
    vector<int> p(ins.k);
    iota(p.begin(), p.end(), 0);
    sort(p.begin(), p.end(), [&](int a, int b) {
        if (ins.deadline[a] != ins.deadline[b]) return ins.deadline[a] < ins.deadline[b];
        return ins.d[0][a + 1] < ins.d[0][b + 1];
    });
    return p;
}

static vector<int> randomPerm(int k, mt19937& rng) {
    vector<int> p(k);
    iota(p.begin(), p.end(), 0);
    shuffle(p.begin(), p.end(), rng);
    return p;
}

static void mutateSwap(vector<int>& p, mt19937& rng) {
    if (p.size() < 2) return;
    uniform_int_distribution<int> dist(0, (int)p.size() - 1);
    int a = dist(rng), b = dist(rng);
    while (b == a) b = dist(rng);
    swap(p[a], p[b]);
}

static void mutateBlockSwap(vector<int>& p, mt19937& rng) {
    int k = (int)p.size(), blocks = (k + 1) / 2;
    if (blocks < 2) {
        mutateSwap(p, rng);
        return;
    }
    uniform_int_distribution<int> bd(0, blocks - 1);
    int A = bd(rng), B = bd(rng);
    while (B == A) B = bd(rng);
    int a1 = 2 * A, b1 = 2 * B;
    int lenA = min(2, k - a1), lenB = min(2, k - b1);
    if (lenA != lenB) {
        mutateSwap(p, rng);
        return;
    }
    for (int t = 0; t < lenA; t++) swap(p[a1 + t], p[b1 + t]);
}

static void mutateReverse(vector<int>& p, mt19937& rng) {
    if (p.size() < 2) return;
    uniform_int_distribution<int> dist(0, (int)p.size() - 1);
    int l = dist(rng), r = dist(rng);
    if (l > r) swap(l, r);
    if (l != r) reverse(p.begin() + l, p.begin() + r + 1);
}

static Solution localImproveSteps(const Instance& ins, Solution start, int steps, mt19937& rng) {
    Solution cur = start, best = start;
    uniform_real_distribution<double> ur(0.0, 1.0);
    for (int it = 0; it < steps; it++) {
        vector<int> cand = cur.perm;
        double r = ur(rng);
        if (r < 0.45) mutateSwap(cand, rng);
        else if (r < 0.75) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);
        Solution nxt = evaluatePerm(ins, cand);
        if (nxt.F <= cur.F) {
            cur = nxt;
            if (cur.F < best.F) best = cur;
        }
    }
    return best;
}

template <class Func>
static pair<Solution, double> timed(Func f) {
    auto t0 = chrono::steady_clock::now();
    Solution sol = f();
    auto t1 = chrono::steady_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();
    return {sol, ms};
}

static bool withinBudget(chrono::steady_clock::time_point start, int budgetMs) {
    return chrono::duration<double, milli>(chrono::steady_clock::now() - start).count() < budgetMs;
}

static Solution localSearchBudget(const Instance& ins, int budgetMs, mt19937& rng) {
    Solution cur = evaluatePerm(ins, eddPerm(ins)), best = cur;
    uniform_real_distribution<double> ur(0.0, 1.0);
    auto start = chrono::steady_clock::now();
    while (withinBudget(start, budgetMs)) {
        vector<int> cand = cur.perm;
        double r = ur(rng);
        if (r < 0.50) mutateSwap(cand, rng);
        else if (r < 0.80) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);
        Solution nxt = evaluatePerm(ins, cand);
        if (nxt.F <= cur.F) {
            cur = nxt;
            if (cur.F < best.F) best = cur;
        }
    }
    return best;
}

static Solution simulatedAnnealingBudget(const Instance& ins, int budgetMs, mt19937& rng) {
    Solution cur = evaluatePerm(ins, eddPerm(ins)), best = cur;
    double T = 200.0;
    uniform_real_distribution<double> ur(0.0, 1.0);
    auto start = chrono::steady_clock::now();
    while (withinBudget(start, budgetMs)) {
        vector<int> cand = cur.perm;
        double r = ur(rng);
        if (r < 0.45) mutateSwap(cand, rng);
        else if (r < 0.80) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);
        Solution nxt = evaluatePerm(ins, cand);
        double delta = nxt.F - cur.F;
        if (delta <= 0 || ur(rng) < exp(-delta / max(1e-12, T))) {
            cur = nxt;
            if (cur.F < best.F) best = cur;
        }
        T = max(1e-4, T * 0.9995);
    }
    return best;
}

static vector<int> completionTimes(const Instance& ins, const vector<int>& perm) {
    vector<int> C(ins.k, 0);
    int t = 0;
    for (int pos = 0; pos < (int)perm.size();) {
        if (pos == (int)perm.size() - 1) {
            int i = perm[pos], di = ins.d[0][i + 1];
            C[i] = t + di;
            t += 2 * di;
            pos++;
        } else {
            int i = perm[pos], j = perm[pos + 1];
            int di = ins.d[0][i + 1], dij = ins.d[i + 1][j + 1], dj = ins.d[0][j + 1];
            C[i] = t + di;
            C[j] = t + di + dij;
            t += di + dij + dj;
            pos += 2;
        }
    }
    return C;
}

static vector<double> targetBadness(const Instance& ins, const vector<int>& perm) {
    vector<int> C = completionTimes(ins, perm);
    vector<int> posOf(ins.k, -1);
    for (int i = 0; i < (int)perm.size(); i++) posOf[perm[i]] = i;
    vector<double> bad(ins.k, 0.0);
    for (int x = 0; x < ins.k; x++) {
        double late = max(0, C[x] - ins.deadline[x]);
        bad[x] = ins.lambda * late + 0.05 * ins.d[0][x + 1] + 0.015 * max(0, posOf[x]);
    }
    return bad;
}

static void erasePositions(vector<int>& base, vector<int> idx, vector<int>& removed) {
    sort(idx.rbegin(), idx.rend());
    idx.erase(unique(idx.begin(), idx.end()), idx.end());
    for (int pos : idx) {
        if (0 <= pos && pos < (int)base.size()) {
            removed.push_back(base[pos]);
            base.erase(base.begin() + pos);
        }
    }
}

static int rouletteSelect(const vector<double>& weight, mt19937& rng) {
    double sum = accumulate(weight.begin(), weight.end(), 0.0);
    if (sum <= 0) return 0;
    uniform_real_distribution<double> ur(0.0, sum);
    double x = ur(rng), acc = 0.0;
    for (int i = 0; i < (int)weight.size(); i++) {
        acc += weight[i];
        if (x <= acc) return i;
    }
    return (int)weight.size() - 1;
}

static pair<vector<int>, vector<int>> destroySolution(const Instance& ins, const vector<int>& perm, int removeCount, int op, mt19937& rng) {
    vector<int> base = perm, removed;
    int n = (int)base.size();
    removeCount = max(1, min(removeCount, max(1, n - 1)));
    uniform_int_distribution<int> posDist(0, n - 1);
    if (op == 0) {
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);
        idx.resize(removeCount);
        erasePositions(base, idx, removed);
    } else if (op == 1) {
        vector<double> bad = targetBadness(ins, base);
        vector<int> items = base;
        sort(items.begin(), items.end(), [&](int a, int b) { return bad[a] > bad[b]; });
        items.resize(removeCount);
        vector<int> idx;
        for (int item : items) idx.push_back((int)(find(base.begin(), base.end(), item) - base.begin()));
        erasePositions(base, idx, removed);
    } else if (op == 2) {
        vector<double> bad = targetBadness(ins, base);
        int seed = *max_element(base.begin(), base.end(), [&](int a, int b) { return bad[a] < bad[b]; });
        vector<int> items = base;
        sort(items.begin(), items.end(), [&](int a, int b) {
            double ra = ins.d[a + 1][seed + 1] + 0.35 * abs(ins.deadline[a] - ins.deadline[seed]);
            double rb = ins.d[b + 1][seed + 1] + 0.35 * abs(ins.deadline[b] - ins.deadline[seed]);
            return ra < rb;
        });
        items.resize(removeCount);
        vector<int> idx;
        for (int item : items) idx.push_back((int)(find(base.begin(), base.end(), item) - base.begin()));
        erasePositions(base, idx, removed);
    } else if (op == 3) {
        int start = posDist(rng);
        vector<int> idx;
        for (int t = 0; t < removeCount; t++) idx.push_back((start + t) % n);
        erasePositions(base, idx, removed);
    } else {
        vector<int> C = completionTimes(ins, base);
        vector<pair<double, int>> blocks;
        for (int pos = 0; pos < n; pos += 2) {
            int a = base[pos];
            double score = max(0, C[a] - ins.deadline[a]) * ins.lambda + 0.02 * ins.d[0][a + 1];
            if (pos + 1 < n) {
                int b = base[pos + 1];
                score += max(0, C[b] - ins.deadline[b]) * ins.lambda + 0.02 * ins.d[0][b + 1];
                score += 0.03 * ins.d[a + 1][b + 1];
            }
            blocks.push_back({-score, pos});
        }
        sort(blocks.begin(), blocks.end());
        vector<int> idx;
        for (auto [negScore, pos] : blocks) {
            if ((int)idx.size() >= removeCount) break;
            idx.push_back(pos);
            if ((int)idx.size() < removeCount && pos + 1 < n) idx.push_back(pos + 1);
        }
        erasePositions(base, idx, removed);
    }
    return {base, removed};
}

struct InsertionChoice {
    int item = -1;
    int pos = 0;
    double bestF = 1e100;
    double secondF = 1e100;
    double regret = 0.0;
};

static InsertionChoice bestInsertionForItem(const Instance& ins, const vector<int>& base, int item) {
    InsertionChoice ch;
    ch.item = item;
    for (int pos = 0; pos <= (int)base.size(); pos++) {
        vector<int> cand = base;
        cand.insert(cand.begin() + pos, item);
        double f = evaluatePerm(ins, cand).F;
        if (f < ch.bestF) {
            ch.secondF = ch.bestF;
            ch.bestF = f;
            ch.pos = pos;
        } else if (f < ch.secondF) {
            ch.secondF = f;
        }
    }
    if (ch.secondF > 5e99) ch.secondF = ch.bestF;
    ch.regret = ch.secondF - ch.bestF;
    return ch;
}

static vector<int> repairInsertion(const Instance& ins, vector<int> base, vector<int> removed, bool useRegret) {
    sort(removed.begin(), removed.end(), [&](int a, int b) {
        double pa = ins.deadline[a] - 0.18 * ins.d[0][a + 1];
        double pb = ins.deadline[b] - 0.18 * ins.d[0][b + 1];
        if (fabs(pa - pb) > 1e-9) return pa < pb;
        return ins.d[0][a + 1] > ins.d[0][b + 1];
    });
    while (!removed.empty()) {
        vector<InsertionChoice> choices;
        for (int item : removed) choices.push_back(bestInsertionForItem(ins, base, item));
        int bestIdx = 0;
        for (int i = 1; i < (int)choices.size(); i++) {
            double scoreA = useRegret ? choices[i].regret - 0.02 * choices[i].bestF : -choices[i].bestF;
            double scoreB = useRegret ? choices[bestIdx].regret - 0.02 * choices[bestIdx].bestF : -choices[bestIdx].bestF;
            if (scoreA > scoreB) bestIdx = i;
        }
        int item = choices[bestIdx].item;
        base.insert(base.begin() + choices[bestIdx].pos, item);
        removed.erase(find(removed.begin(), removed.end(), item));
    }
    return base;
}

static Solution lnsBudget(const Instance& ins, int budgetMs, mt19937& rng, const LnsOptions& opt) {
    Solution cur = evaluatePerm(ins, eddPerm(ins));
    Solution randomStart = evaluatePerm(ins, randomPerm(ins.k, rng));
    if (randomStart.F < cur.F) cur = randomStart;
    if (opt.localImprove) cur = localImproveSteps(ins, cur, 12, rng);
    Solution best = cur;

    vector<double> weight = {1.0, 1.4, 1.4, 0.9, 1.2};
    vector<double> score(weight.size(), 0.0), used(weight.size(), 0.0);
    uniform_real_distribution<double> ur(0.0, 1.0);
    double T = 120.0;
    int iter = 0;
    auto start = chrono::steady_clock::now();
    while (withinBudget(start, budgetMs)) {
        int op = rouletteSelect(weight, rng);
        used[op] += 1.0;
        int removeCount = max(2, min(ins.k - 1, (int)round(ins.k * opt.destroyRatio * (0.80 + 0.55 * ur(rng)))));
        if (ur(rng) < 0.10) removeCount = max(removeCount, (int)round(ins.k * 0.42));
        auto [base, removed] = destroySolution(ins, cur.perm, removeCount, op, rng);
        if (removed.empty()) continue;
        Solution nxt = evaluatePerm(ins, repairInsertion(ins, base, removed, opt.useRegret));
        if (opt.localImprove && (nxt.F <= cur.F || ur(rng) < 0.25)) {
            nxt = localImproveSteps(ins, nxt, 18, rng);
        }
        double delta = nxt.F - cur.F;
        bool accepted = false;
        if (delta <= 0 || (opt.acceptWorse && ur(rng) < exp(-delta / max(1e-12, T)))) {
            cur = nxt;
            accepted = true;
        }
        if (nxt.F < best.F) {
            best = nxt;
            cur = nxt;
            score[op] += 9.0;
        } else if (accepted && delta < 0) {
            score[op] += 3.0;
        } else if (accepted) {
            score[op] += 0.8;
        }
        iter++;
        if (opt.adaptiveWeights && iter % 35 == 0) {
            for (int i = 0; i < (int)weight.size(); i++) {
                double avg = used[i] > 0 ? score[i] / used[i] : 0.0;
                weight[i] = 0.80 * weight[i] + 0.20 * max(0.15, avg);
                score[i] = used[i] = 0.0;
            }
        }
        T = max(1e-4, T * 0.9965);
    }
    return best;
}

static string variantName(const LnsOptions& opt) {
    if (opt.useRegret && opt.adaptiveWeights && opt.localImprove && opt.acceptWorse && fabs(opt.destroyRatio - 0.30) < 1e-9) return "full";
    if (!opt.useRegret) return "no_regret";
    if (!opt.adaptiveWeights) return "no_adaptive_weight";
    if (!opt.localImprove) return "no_local_search";
    if (!opt.acceptWorse) return "no_probability_acceptance";
    ostringstream ss;
    ss << "destroy_ratio_" << fixed << setprecision(2) << opt.destroyRatio;
    return ss.str();
}

static void writePerInstance(const fs::path& path, const vector<ResultRow>& rows) {
    ofstream f(path);
    f << "experiment,k,instance_id,run_id,instance_seed,run_seed,algorithm,variant,budget_ms,time_ms,F,D,tardiness,gap_to_best_pct\n";
    f << fixed << setprecision(6);
    for (const auto& r : rows) {
        f << csvEscape(r.experiment) << "," << r.k << "," << r.instanceId << "," << r.runId << ","
          << r.instanceSeed << "," << r.runSeed << ","
          << csvEscape(r.algorithm) << "," << csvEscape(r.variant) << "," << r.budgetMs << "," << r.timeMs << ","
          << r.F << "," << r.D << "," << r.tardiness << "," << r.gapToBestPct << "\n";
    }
}

static vector<SummaryRow> summarizeRows(const vector<ResultRow>& rows) {
    map<tuple<string, int, string, string>, vector<const ResultRow*>> groups;
    map<tuple<string, int, int, int>, double> bestByInstance;
    for (const auto& r : rows) {
        groups[{r.experiment, r.k, r.algorithm, r.variant}].push_back(&r);
        auto key = make_tuple(r.experiment, r.k, r.instanceId, r.runId);
        if (!bestByInstance.count(key) || r.F < bestByInstance[key]) bestByInstance[key] = r.F;
    }
    vector<SummaryRow> out;
    for (auto& [key, vals] : groups) {
        SummaryRow s;
        tie(s.experiment, s.k, s.algorithm, s.variant) = key;
        s.count = (int)vals.size();
        vector<double> F, D, tard, tm, gap;
        int wins = 0;
        for (const ResultRow* r : vals) {
            F.push_back(r->F);
            D.push_back(r->D);
            tard.push_back(r->tardiness);
            tm.push_back(r->timeMs);
            double best = bestByInstance[{r->experiment, r->k, r->instanceId, r->runId}];
            gap.push_back((r->F - best) / max(1.0, best) * 100.0);
            if (fabs(r->F - best) < 1e-9) wins++;
            s.budgetMs = r->budgetMs;
        }
        s.meanF = mean(F);
        s.stdF = stdev(F);
        s.meanD = mean(D);
        s.meanTardiness = mean(tard);
        s.meanTimeMs = mean(tm);
        s.meanGapToBestPct = mean(gap);
        s.winRatePct = 100.0 * wins / max(1, s.count);
        out.push_back(s);
    }
    sort(out.begin(), out.end(), [](const SummaryRow& a, const SummaryRow& b) {
        return tie(a.experiment, a.k, a.algorithm, a.variant) < tie(b.experiment, b.k, b.algorithm, b.variant);
    });
    return out;
}

static void writeSummary(const fs::path& path, const vector<SummaryRow>& rows) {
    ofstream f(path);
    f << "experiment,k,algorithm,variant,count,budget_ms,mean_F,std_F,mean_D,mean_tardiness,mean_time_ms,mean_gap_to_best_pct,win_rate_pct\n";
    f << fixed << setprecision(6);
    for (const auto& s : rows) {
        f << csvEscape(s.experiment) << "," << s.k << "," << csvEscape(s.algorithm) << "," << csvEscape(s.variant) << ","
          << s.count << "," << s.budgetMs << "," << s.meanF << "," << s.stdF << "," << s.meanD << ","
          << s.meanTardiness << "," << s.meanTimeMs << "," << s.meanGapToBestPct << "," << s.winRatePct << "\n";
    }
}

static void writeLatexTable(const fs::path& path, const vector<SummaryRow>& rows, const string& experiment) {
    ofstream f(path);
    f << "\\begin{tabular}{rrrrrr}\n\\toprule\n";
    f << "$k$ & " << (experiment == "ablation" ? "Variant" : "Algorithm") << " & Mean $F$ & Std. $F$ & Gap (\\%) & Win (\\%) \\\\\n\\midrule\n";
    f << fixed << setprecision(2);
    for (const auto& s : rows) {
        if (s.experiment != experiment) continue;
        string name = s.algorithm == "LNS" && s.variant != "full" ? s.variant : s.algorithm;
        f << s.k << " & " << latexEscape(name) << " & " << s.meanF << " & " << s.stdF << " & "
          << s.meanGapToBestPct << " & " << s.winRatePct << " \\\\\n";
    }
    f << "\\bottomrule\n\\end{tabular}\n";
}

static void writeSimpleSvgLine(const fs::path& path, const vector<SummaryRow>& rows, const string& experiment) {
    vector<string> algs = {"LocalSearch", "SA", "LNS"};
    map<pair<string, int>, double> y;
    vector<int> ks;
    for (const auto& s : rows) {
        if (s.experiment == experiment && s.variant == "full") {
            y[{s.algorithm, s.k}] = s.meanF;
            ks.push_back(s.k);
        }
    }
    sort(ks.begin(), ks.end());
    ks.erase(unique(ks.begin(), ks.end()), ks.end());
    if (ks.empty()) return;
    double ymin = 1e100, ymax = -1e100;
    for (auto& kv : y) ymin = min(ymin, kv.second), ymax = max(ymax, kv.second);
    double pad = max(1.0, (ymax - ymin) * 0.08);
    ymin -= pad; ymax += pad;
    int W = 900, H = 560, L = 80, R = 30, T = 35, B = 70;
    auto sx = [&](int k) {
        if (ks.size() == 1) return (double)(L + (W - L - R) / 2);
        return L + (double)(k - ks.front()) / (ks.back() - ks.front()) * (W - L - R);
    };
    auto sy = [&](double v) { return T + (ymax - v) / (ymax - ymin) * (H - T - B); };
    vector<string> colors = {"#7a7a7a", "#2f75b5", "#c0392b"};
    ofstream f(path);
    f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H << "\" viewBox=\"0 0 " << W << " " << H << "\">\n";
    f << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"450\" y=\"24\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"18\">Unified Time Budget: Mean Objective by k</text>\n";
    f << "<line x1=\"" << L << "\" y1=\"" << H-B << "\" x2=\"" << W-R << "\" y2=\"" << H-B << "\" stroke=\"#333\"/><line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L << "\" y2=\"" << H-B << "\" stroke=\"#333\"/>\n";
    for (int k : ks) {
        f << "<text x=\"" << sx(k) << "\" y=\"" << H-B+28 << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"13\">" << k << "</text>\n";
    }
    f << "<text x=\"450\" y=\"535\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">k</text><text transform=\"translate(20 280) rotate(-90)\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">Mean F</text>\n";
    for (int ai = 0; ai < (int)algs.size(); ai++) {
        string alg = algs[ai];
        f << "<polyline fill=\"none\" stroke=\"" << colors[ai] << "\" stroke-width=\"2.5\" points=\"";
        for (int k : ks) if (y.count({alg, k})) f << sx(k) << "," << sy(y[{alg, k}]) << " ";
        f << "\"/>\n";
        for (int k : ks) if (y.count({alg, k})) f << "<circle cx=\"" << sx(k) << "\" cy=\"" << sy(y[{alg, k}]) << "\" r=\"4\" fill=\"" << colors[ai] << "\"/>\n";
        f << "<rect x=\"" << (690) << "\" y=\"" << (55 + ai * 24) << "\" width=\"14\" height=\"14\" fill=\"" << colors[ai] << "\"/><text x=\"712\" y=\"" << (67 + ai * 24) << "\" font-family=\"Arial\" font-size=\"13\">" << alg << "</text>\n";
    }
    f << "</svg>\n";
}

static void writeAblationSvg(const fs::path& path, const vector<SummaryRow>& rows) {
    vector<SummaryRow> vals;
    for (const auto& s : rows) if (s.experiment == "ablation" && s.k == 32) vals.push_back(s);
    if (vals.empty()) return;
    sort(vals.begin(), vals.end(), [](const SummaryRow& a, const SummaryRow& b) { return a.meanF < b.meanF; });
    int W = 980, H = 560, L = 210, R = 50, T = 40, B = 45;
    double ymax = 0;
    for (auto& s : vals) ymax = max(ymax, s.meanF);
    ofstream f(path);
    f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H << "\" viewBox=\"0 0 " << W << " " << H << "\">\n";
    f << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"490\" y=\"24\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"18\">Ablation Mean Objective (k=32)</text>\n";
    int n = (int)vals.size();
    double barH = (H - T - B) / max(1, n) * 0.62;
    for (int i = 0; i < n; i++) {
        double y = T + i * (H - T - B) / max(1, n) + 8;
        double w = vals[i].meanF / ymax * (W - L - R);
        f << "<text x=\"" << L-8 << "\" y=\"" << y + barH*0.7 << "\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"12\">" << vals[i].variant << "</text>\n";
        f << "<rect x=\"" << L << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << barH << "\" fill=\"#4c78a8\"/>\n";
        f << "<text x=\"" << L+w+5 << "\" y=\"" << y + barH*0.7 << "\" font-family=\"Arial\" font-size=\"12\">" << fixed << setprecision(1) << vals[i].meanF << "</text>\n";
    }
    f << "</svg>\n";
}

static Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto need = [&](int& i) -> string {
            if (i + 1 >= argc) throw runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--instances") cfg.instancesPerK = stoi(need(i));
        else if (a == "--budget-ms") cfg.budgetMs = stoi(need(i));
        else if (a == "--repeats") cfg.repeatsPerInstance = stoi(need(i));
        else if (a == "--seed") cfg.baseSeed = (unsigned)stoul(need(i));
        else if (a == "--out") cfg.outDir = need(i);
        else if (a == "--k") {
            cfg.kValues.clear();
            string s = need(i), cur;
            stringstream ss(s);
            while (getline(ss, cur, ',')) if (!cur.empty()) cfg.kValues.push_back(stoi(cur));
        } else if (a == "--help") {
            cout << "Usage: improved_experiments.exe [--instances 20] [--repeats 3] [--budget-ms 300] [--seed 20260524] [--k 16,24,32,40,48,64] [--out results]\n";
            exit(0);
        }
    }
    return cfg;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Config cfg = parseArgs(argc, argv);
    fs::path out = cfg.outDir;
    fs::path csvDir = out / "csv";
    fs::create_directories(out);
    fs::create_directories(csvDir);
    fs::create_directories(out / "figures");
    fs::create_directories(out / "latex_tables");

    vector<ResultRow> perRows;
    vector<ResultRow> ablationRows;
    vector<LnsOptions> ablations;
    ablations.push_back({true, true, true, true, 0.30});
    ablations.push_back({false, true, true, true, 0.30});
    ablations.push_back({true, false, true, true, 0.30});
    ablations.push_back({true, true, false, true, 0.30});
    ablations.push_back({true, true, true, false, 0.30});
    ablations.push_back({true, true, true, true, 0.15});
    ablations.push_back({true, true, true, true, 0.45});

    cout << "Improved experiments: instancesPerK=" << cfg.instancesPerK << ", repeatsPerInstance=" << cfg.repeatsPerInstance << ", budgetMs=" << cfg.budgetMs << "\n";
    for (int k : cfg.kValues) {
        cout << "k=" << k << "\n";
        for (int inst = 0; inst < cfg.instancesPerK; inst++) {
            unsigned instanceSeed = cfg.baseSeed + (unsigned)k * 100000u + (unsigned)inst * 97u;
            Instance ins = generateInstance(cfg, k, instanceSeed);
            for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                unsigned runSeedBase = cfg.baseSeed + (unsigned)k * 10000u + (unsigned)inst * 101u + (unsigned)rep * 1009u;
                vector<ResultRow> current;
            auto add = [&](const string& experiment, const string& alg, const string& variant, Solution sol, double timeMs) {
                current.push_back({experiment, k, inst, rep, instanceSeed, runSeedBase, alg, variant, (double)cfg.budgetMs, timeMs, sol.F, sol.D, sol.tardiness, 0.0});
            };
            {
                mt19937 rng(runSeedBase + 11u);
                auto [sol, ms] = timed([&] { return localSearchBudget(ins, cfg.budgetMs, rng); });
                add("budget", "LocalSearch", "full", sol, ms);
            }
            {
                mt19937 rng(runSeedBase + 23u);
                auto [sol, ms] = timed([&] { return simulatedAnnealingBudget(ins, cfg.budgetMs, rng); });
                add("budget", "SA", "full", sol, ms);
            }
            {
                mt19937 rng(runSeedBase + 37u);
                auto [sol, ms] = timed([&] { return lnsBudget(ins, cfg.budgetMs, rng, LnsOptions{}); });
                add("budget", "LNS", "full", sol, ms);
            }
            double bestF = 1e100;
            for (auto& r : current) bestF = min(bestF, r.F);
            for (auto& r : current) {
                r.gapToBestPct = (r.F - bestF) / max(1.0, bestF) * 100.0;
                perRows.push_back(r);
            }
            if (k == 32) {
                vector<ResultRow> curAbl;
                for (const auto& opt : ablations) {
                    mt19937 rng(runSeedBase + 1009u + (unsigned)(opt.destroyRatio * 1000) + (opt.useRegret ? 1u : 101u) + (opt.adaptiveWeights ? 3u : 103u) + (opt.localImprove ? 5u : 105u) + (opt.acceptWorse ? 7u : 107u));
                    auto [sol, ms] = timed([&] { return lnsBudget(ins, cfg.budgetMs, rng, opt); });
                    curAbl.push_back({"ablation", k, inst, rep, instanceSeed, runSeedBase, "LNS", variantName(opt), (double)cfg.budgetMs, ms, sol.F, sol.D, sol.tardiness, 0.0});
                }
                double bestA = 1e100;
                for (auto& r : curAbl) bestA = min(bestA, r.F);
                for (auto& r : curAbl) {
                    r.gapToBestPct = (r.F - bestA) / max(1.0, bestA) * 100.0;
                    ablationRows.push_back(r);
                }
            }
            }
        }
    }

    vector<ResultRow> allRows = perRows;
    allRows.insert(allRows.end(), ablationRows.begin(), ablationRows.end());
    vector<SummaryRow> summary = summarizeRows(allRows);

    writePerInstance(csvDir / "per_instance_results.csv", perRows);
    writePerInstance(csvDir / "ablation_results.csv", ablationRows);
    writeSummary(csvDir / "summary.csv", summary);
    writeLatexTable(out / "latex_tables" / "budget_summary.tex", summary, "budget");
    writeLatexTable(out / "latex_tables" / "ablation_summary.tex", summary, "ablation");
    writeSimpleSvgLine(out / "figures" / "budget_mean_objective_by_k.svg", summary, "budget");
    writeAblationSvg(out / "figures" / "ablation_k32.svg", summary);

    ofstream meta(out / "experiment_config.txt");
    meta << "instances_per_k=" << cfg.instancesPerK << "\n";
    meta << "repeats_per_instance=" << cfg.repeatsPerInstance << "\n";
    meta << "budget_ms=" << cfg.budgetMs << "\n";
    meta << "base_seed=" << cfg.baseSeed << "\n";
    meta << "k_values=";
    for (int i = 0; i < (int)cfg.kValues.size(); i++) meta << (i ? "," : "") << cfg.kValues[i];
    meta << "\n";
    meta << "outputs=csv/summary.csv,csv/per_instance_results.csv,csv/ablation_results.csv,figures/,latex_tables/\n";

    cout << "Done. Results written to " << out.string() << "\n";
    return 0;
}
