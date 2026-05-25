/*
 * experiments.cpp
 *
 * 中文实验程序：以大邻域搜索（Large Neighborhood Search, LNS）为主方法，
 * 研究较大目标数下带时间惩罚的双载荷网格配送优化问题。
 *
 * 编译：
 *   g++ -std=c++17 -O2 experiments.cpp -o experiments
 * Windows cmd：
 *   g++ -std=c++17 -O2 experiments.cpp -o experiments.exe
 *   experiments.exe
 *
 * 输出：
 *   table1_overall.csv
 *   table2_by_k.csv
 *   table3_by_beta.csv
 *   table4_by_lambda.csv
 *   table5_ablation.csv
 *   table6_extended_k.csv
 *   table7_exact_gap.csv
 *   table8_paired_results.csv
 */

#include <bits/stdc++.h>
using namespace std;

static const int INF = 1e9;

struct Config {
    int n = 40;
    int m = 40;
    double obstacleRate = 0.18;

    vector<int> kValues = {16, 24, 32, 40, 48};
    vector<int> extendedKValues = {32, 40, 48};
    vector<double> betaValues = {1.2, 1.6, 2.0, 2.4};
    vector<double> lambdaValues = {0.5, 1.0, 2.0, 3.0};

    int instancesPerGroup = 20;
    int repeatsPerInstance = 3;
    int extendedInstancesPerGroup = 20;
    int exactMatchingLimit = 16;

    vector<int> exactKValues = {6, 8};
    int exactInstancesPerGroup = 10;
    int exactPermutationLimit = 10;

    int localIterations = 500;
    int saIterations = 1400;
    int lnsIterations = 320;
    int lnsRepairCandidates = 0; // 0 表示修复阶段枚举所有插入位置，提升 LNS 强度
    double lnsDestroyRatio = 0.34;
    double lnsT0 = 120.0;
    double lnsAlpha = 0.9965;
    int lnsSegmentLength = 40;
    int lnsStagnationLimit = 90;
    int lnsPolishIterations = 24;

    double saT0 = 200.0;
    double saTmin = 1e-4;
    double saAlpha = 0.9995;

    unsigned seed = 20260515;
};

struct Point { int x, y; };

struct Instance {
    int n = 0, m = 0, k = 0;
    vector<string> grid;
    Point S{};
    vector<Point> X;
    vector<vector<int>> d; // 0 为仓库，i+1 为目标 i
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

struct Stats {
    double F = 0, D = 0, tardiness = 0, gap = 0, timeMs = 0, stdF = 0;
    int count = 0;
};

struct RunValues {
    vector<double> F, D, tard, gap, timeMs;
};

double meanOf(const vector<double>& a) {
    if (a.empty()) return 0.0;
    double s = accumulate(a.begin(), a.end(), 0.0);
    return s / (double)a.size();
}

double stdOf(const vector<double>& a) {
    if (a.size() <= 1) return 0.0;
    double mu = meanOf(a), s = 0;
    for (double x : a) s += (x - mu) * (x - mu);
    return sqrt(s / (double)(a.size() - 1));
}

Stats summarize(const RunValues& rv) {
    Stats st;
    st.F = meanOf(rv.F);
    st.D = meanOf(rv.D);
    st.tardiness = meanOf(rv.tard);
    st.gap = meanOf(rv.gap);
    st.timeMs = meanOf(rv.timeMs);
    st.stdF = stdOf(rv.F);
    st.count = (int)rv.F.size();
    return st;
}

vector<vector<int>> bfsGrid(const vector<string>& grid, Point src) {
    int n = (int)grid.size(), m = (int)grid[0].size();
    vector<vector<int>> dist(n, vector<int>(m, INF));
    queue<Point> q;
    dist[src.x][src.y] = 0;
    q.push(src);
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        Point cur = q.front(); q.pop();
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

bool computeDistances(Instance& ins) {
    int k = ins.k;
    ins.d.assign(k + 1, vector<int>(k + 1, INF));
    vector<Point> pts;
    pts.push_back(ins.S);
    for (Point p : ins.X) pts.push_back(p);
    for (int i = 0; i <= k; i++) {
        auto dist = bfsGrid(ins.grid, pts[i]);
        for (int j = 0; j <= k; j++) ins.d[i][j] = dist[pts[j].x][pts[j].y];
    }
    for (int i = 1; i <= k; i++) if (ins.d[0][i] >= INF) return false;
    return true;
}

Instance generateInstance(int n, int m, int k, double obstacleRate, double beta, double lambda, mt19937& rng) {
    uniform_real_distribution<double> ur(0.0, 1.0);
    uniform_int_distribution<int> rx(0, n - 1), ry(0, m - 1);
    while (true) {
        Instance ins;
        ins.n = n; ins.m = m; ins.k = k; ins.beta = beta; ins.lambda = lambda;
        ins.grid.assign(n, string(m, '.'));
        for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) if (ur(rng) < obstacleRate) ins.grid[i][j] = '#';
        ins.S = {rx(rng), ry(rng)};
        ins.grid[ins.S.x][ins.S.y] = 'S';
        set<pair<int,int>> used;
        used.insert({ins.S.x, ins.S.y});
        int attempts = 0;
        while ((int)ins.X.size() < k && attempts < n * m * 30) {
            attempts++;
            int x = rx(rng), y = ry(rng);
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
            int tau = max(1, (int)round(beta * di) + epsDist(rng));
            ins.deadline[i] = tau;
        }
        return ins;
    }
}

Solution evaluatePerm(const Instance& ins, const vector<int>& perm) {
    Solution sol;
    sol.perm = perm;
    int t = 0, D = 0, tard = 0;
    int k = (int)perm.size();
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
            int i = perm[pos], j = perm[pos + 1];
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
    sol.D = D; sol.tardiness = tard; sol.F = (double)D + ins.lambda * tard;
    return sol;
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
    vector<int> rem(ins.k), perm;
    iota(rem.begin(), rem.end(), 0);
    int t = 0;
    while (!rem.empty()) {
        int first = -1;
        double best = 1e100;
        for (int a : rem) {
            double late = max(0, t + ins.d[0][a + 1] - ins.deadline[a]);
            double score = ins.d[0][a + 1] + ins.lambda * late + 0.12 * ins.deadline[a];
            if (score < best) best = score, first = a;
        }
        perm.push_back(first);
        rem.erase(find(rem.begin(), rem.end(), first));
        if (rem.empty()) break;
        int second = -1; best = 1e100;
        for (int b : rem) {
            double late = max(0, t + ins.d[0][first + 1] + ins.d[first + 1][b + 1] - ins.deadline[b]);
            double score = ins.d[first + 1][b + 1] + ins.d[0][b + 1] + ins.lambda * late + 0.08 * ins.deadline[b];
            if (score < best) best = score, second = b;
        }
        perm.push_back(second);
        rem.erase(find(rem.begin(), rem.end(), second));
        t += ins.d[0][first + 1] + ins.d[first + 1][second + 1] + ins.d[0][second + 1];
    }
    return perm;
}

struct MatchingResult { int lb = INF; vector<pair<int,int>> pairs; };

MatchingResult minDistanceMatchingDP(const Instance& ins) {
    int k = ins.k;
    int N = 1 << k;
    vector<int> dp(N, INF);
    vector<pair<int,int>> choice(N, {-1, -1});
    dp[0] = 0;
    for (int mask = 1; mask < N; mask++) {
        int i = -1;
        for (int x = 0; x < k; x++) if (mask & (1 << x)) { i = x; break; }
        int cnt = __builtin_popcount((unsigned)mask);
        if (cnt % 2 == 1) {
            int nm = mask ^ (1 << i);
            int cost = 2 * ins.d[0][i + 1];
            if (dp[nm] + cost < dp[mask]) dp[mask] = dp[nm] + cost, choice[mask] = {i, -1};
        }
        for (int j = i + 1; j < k; j++) if (mask & (1 << j)) {
            int nm = mask ^ (1 << i) ^ (1 << j);
            int cost = ins.d[0][i + 1] + ins.d[i + 1][j + 1] + ins.d[0][j + 1];
            if (dp[nm] + cost < dp[mask]) dp[mask] = dp[nm] + cost, choice[mask] = {i, j};
        }
    }
    MatchingResult res; res.lb = dp[N - 1];
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
    vector<int> rem(ins.k);
    iota(rem.begin(), rem.end(), 0);
    MatchingResult res; res.lb = 0;
    while ((int)rem.size() >= 2) {
        int bx = -1, by = -1, best = INF;
        for (int x = 0; x < (int)rem.size(); x++) for (int y = x + 1; y < (int)rem.size(); y++) {
            int a = rem[x], b = rem[y];
            int cost = ins.d[0][a + 1] + ins.d[a + 1][b + 1] + ins.d[0][b + 1];
            if (cost < best) best = cost, bx = x, by = y;
        }
        int a = rem[bx], b = rem[by];
        res.pairs.push_back({a, b}); res.lb += best;
        if (bx > by) swap(bx, by);
        rem.erase(rem.begin() + by); rem.erase(rem.begin() + bx);
    }
    if (!rem.empty()) { int a = rem[0]; res.pairs.push_back({a, -1}); res.lb += 2 * ins.d[0][a + 1]; }
    return res;
}

MatchingResult distanceReferenceMatching(const Instance& ins, int exactLimit) {
    if (ins.k <= exactLimit) return minDistanceMatchingDP(ins);
    return greedyDistanceMatching(ins);
}

vector<int> matchingEDDPerm(const Instance& ins) {
    auto mr = distanceReferenceMatching(ins, 16);
    vector<pair<int, pair<int,int>>> blocks;
    for (auto [a, b] : mr.pairs) {
        int key = (b == -1 ? ins.deadline[a] : min(ins.deadline[a], ins.deadline[b]));
        blocks.push_back({key, {a, b}});
    }
    sort(blocks.begin(), blocks.end());
    vector<int> perm;
    for (auto item : blocks) {
        int a = item.second.first, b = item.second.second;
        if (b == -1) perm.push_back(a);
        else {
            if (ins.deadline[b] < ins.deadline[a]) swap(a, b);
            perm.push_back(a); perm.push_back(b);
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
    int k = (int)p.size(), blocks = (k + 1) / 2;
    if (blocks < 2) { mutateSwap(p, rng); return; }
    uniform_int_distribution<int> bd(0, blocks - 1);
    int A = bd(rng), B = bd(rng);
    while (B == A) B = bd(rng);
    int a1 = 2 * A, b1 = 2 * B;
    int lenA = min(2, k - a1), lenB = min(2, k - b1);
    if (lenA != lenB) { mutateSwap(p, rng); return; }
    for (int t = 0; t < lenA; t++) swap(p[a1 + t], p[b1 + t]);
}

void mutateReverse(vector<int>& p, mt19937& rng) {
    if (p.size() < 2) return;
    uniform_int_distribution<int> dist(0, (int)p.size() - 1);
    int l = dist(rng), r = dist(rng);
    if (l > r) swap(l, r);
    if (l != r) reverse(p.begin() + l, p.begin() + r + 1);
}

Solution localSearch(const Instance& ins, vector<int> start, int iterations, mt19937& rng) {
    Solution cur = evaluatePerm(ins, start), best = cur;
    uniform_real_distribution<double> ur(0.0, 1.0);
    for (int it = 0; it < iterations; it++) {
        vector<int> cand = cur.perm;
        double r = ur(rng);
        if (r < 0.50) mutateSwap(cand, rng);
        else if (r < 0.80) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);
        Solution nxt = evaluatePerm(ins, cand);
        if (nxt.F <= cur.F) { cur = nxt; if (cur.F < best.F) best = cur; }
    }
    return best;
}

Solution simulatedAnnealing(const Instance& ins, vector<int> start, const Config& cfg, mt19937& rng) {
    Solution cur = evaluatePerm(ins, start), best = cur;
    double T = cfg.saT0;
    uniform_real_distribution<double> ur(0.0, 1.0);
    for (int it = 0; it < cfg.saIterations && T > cfg.saTmin; it++) {
        vector<int> cand = cur.perm;
        double r = ur(rng);
        if (r < 0.45) mutateSwap(cand, rng);
        else if (r < 0.80) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);
        Solution nxt = evaluatePerm(ins, cand);
        double delta = nxt.F - cur.F;
        if (delta <= 0 || ur(rng) < exp(-delta / max(1e-12, T))) {
            cur = nxt; if (cur.F < best.F) best = cur;
        }
        T *= cfg.saAlpha;
    }
    return best;
}


vector<int> completionTimes(const Instance& ins, const vector<int>& perm) {
    vector<int> C(ins.k, 0);
    int t = 0;
    int k = (int)perm.size();
    for (int pos = 0; pos < k; ) {
        if (pos == k - 1) {
            int i = perm[pos];
            int di = ins.d[0][i + 1];
            C[i] = t + di;
            t += 2 * di;
            pos += 1;
        } else {
            int i = perm[pos], j = perm[pos + 1];
            int di = ins.d[0][i + 1];
            int dij = ins.d[i + 1][j + 1];
            int dj = ins.d[0][j + 1];
            C[i] = t + di;
            C[j] = t + di + dij;
            t += di + dij + dj;
            pos += 2;
        }
    }
    return C;
}

vector<double> targetBadness(const Instance& ins, const vector<int>& perm) {
    vector<int> C = completionTimes(ins, perm);
    vector<int> posOf(ins.k, -1);
    for (int i = 0; i < (int)perm.size(); i++) posOf[perm[i]] = i;
    vector<double> bad(ins.k, 0.0);
    for (int x = 0; x < ins.k; x++) {
        double late = max(0, C[x] - ins.deadline[x]);
        double dist = ins.d[0][x + 1];
        double posPenalty = (posOf[x] < 0 ? 0.0 : 0.015 * posOf[x]);
        bad[x] = ins.lambda * late + 0.05 * dist + posPenalty;
    }
    return bad;
}

void erasePositions(vector<int>& base, const vector<int>& positions, vector<int>& removed) {
    vector<int> idx = positions;
    sort(idx.rbegin(), idx.rend());
    idx.erase(unique(idx.begin(), idx.end()), idx.end());
    for (int pos : idx) {
        if (0 <= pos && pos < (int)base.size()) {
            removed.push_back(base[pos]);
            base.erase(base.begin() + pos);
        }
    }
}

int rouletteSelect(const vector<double>& w, mt19937& rng) {
    double sum = accumulate(w.begin(), w.end(), 0.0);
    if (sum <= 0) return 0;
    uniform_real_distribution<double> ur(0.0, sum);
    double x = ur(rng), acc = 0;
    for (int i = 0; i < (int)w.size(); i++) {
        acc += w[i];
        if (x <= acc) return i;
    }
    return (int)w.size() - 1;
}

pair<vector<int>, vector<int>> destroySolution(
    const Instance& ins,
    const vector<int>& perm,
    int removeCount,
    int op,
    mt19937& rng
) {
    vector<int> base = perm, removed;
    int n = (int)base.size();
    if (n <= 1) return {base, removed};
    removeCount = min(removeCount, n - 1);
    uniform_int_distribution<int> posDist(0, n - 1);

    if (op == 0) { // 随机删除：保持探索性
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);
        idx.resize(removeCount);
        erasePositions(base, idx, removed);
    } else if (op == 1) { // 最坏删除：优先移除迟到贡献大的目标
        vector<double> bad = targetBadness(ins, base);
        vector<int> items = base;
        sort(items.begin(), items.end(), [&](int a, int b) {
            return bad[a] > bad[b];
        });
        items.resize(removeCount);
        vector<int> idx;
        for (int item : items) {
            auto it = find(base.begin(), base.end(), item);
            if (it != base.end()) idx.push_back((int)(it - base.begin()));
        }
        erasePositions(base, idx, removed);
    } else if (op == 2) { // 相关删除：围绕一个种子移除空间/时间相近任务
        vector<double> bad = targetBadness(ins, base);
        int seed;
        if (uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.70) {
            seed = max_element(base.begin(), base.end(), [&](int a, int b){ return bad[a] < bad[b]; })[0];
        } else seed = base[posDist(rng)];
        vector<int> items = base;
        sort(items.begin(), items.end(), [&](int a, int b) {
            double ra = ins.d[a + 1][seed + 1] + 0.35 * abs(ins.deadline[a] - ins.deadline[seed]);
            double rb = ins.d[b + 1][seed + 1] + 0.35 * abs(ins.deadline[b] - ins.deadline[seed]);
            return ra < rb;
        });
        items.resize(removeCount);
        vector<int> idx;
        for (int item : items) {
            auto it = find(base.begin(), base.end(), item);
            if (it != base.end()) idx.push_back((int)(it - base.begin()));
        }
        erasePositions(base, idx, removed);
    } else if (op == 3) { // 连续块删除：重排局部配送块顺序
        int start = posDist(rng);
        vector<int> idx;
        for (int t = 0; t < removeCount; t++) idx.push_back((start + t) % n);
        erasePositions(base, idx, removed);
    } else { // 配送块坏块删除：移除若干表现差的双载荷块
        vector<int> C = completionTimes(ins, base);
        vector<pair<double,int>> blocks;
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

InsertionChoice bestInsertionForItem(const Instance& ins, const vector<int>& base, int item) {
    InsertionChoice ch;
    ch.item = item;
    int L = (int)base.size();
    for (int pos = 0; pos <= L; pos++) {
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

vector<int> repairByRegretInsertion(const Instance& ins, vector<int> base, vector<int> removed, mt19937& rng) {
    // 先处理更紧急、更远、更可能造成迟到的任务，降低贪心修复的短视性。
    sort(removed.begin(), removed.end(), [&](int a, int b) {
        double pa = ins.deadline[a] - 0.18 * ins.d[0][a + 1];
        double pb = ins.deadline[b] - 0.18 * ins.d[0][b + 1];
        if (fabs(pa - pb) > 1e-9) return pa < pb;
        return ins.d[0][a + 1] > ins.d[0][b + 1];
    });

    while (!removed.empty()) {
        vector<InsertionChoice> choices;
        choices.reserve(removed.size());
        for (int item : removed) choices.push_back(bestInsertionForItem(ins, base, item));
        // regret-2：优先插入“如果不现在插会损失很大”的任务；并兼顾直接目标值。
        int bestIdx = 0;
        for (int i = 1; i < (int)choices.size(); i++) {
            double scoreA = choices[i].regret - 0.02 * choices[i].bestF;
            double scoreB = choices[bestIdx].regret - 0.02 * choices[bestIdx].bestF;
            if (scoreA > scoreB) bestIdx = i;
        }
        int item = choices[bestIdx].item;
        int pos = choices[bestIdx].pos;
        base.insert(base.begin() + pos, item);
        removed.erase(find(removed.begin(), removed.end(), item));
    }
    return base;
}

Solution shortIntensification(const Instance& ins, Solution start, int iterations, mt19937& rng) {
    Solution cur = start, best = start;
    uniform_real_distribution<double> ur(0.0, 1.0);
    for (int it = 0; it < iterations; it++) {
        vector<int> cand = cur.perm;
        double r = ur(rng);
        if (r < 0.35) mutateSwap(cand, rng);
        else if (r < 0.65) mutateBlockSwap(cand, rng);
        else mutateReverse(cand, rng);
        Solution nxt = evaluatePerm(ins, cand);
        if (nxt.F <= cur.F) {
            cur = nxt;
            if (cur.F < best.F) best = cur;
        }
    }
    return best;
}



// 确定性强化：在若干轮内系统枚举交换、反转、重插入和双载荷块操作。
// 该步骤只嵌入 LNS，用于把“破坏--修复”得到的粗解继续压到局部极小。
Solution deterministicPolish(const Instance& ins, Solution start, int maxRounds) {
    Solution cur = start;
    int k = (int)cur.perm.size();
    if (k <= 1) return cur;

    for (int round = 0; round < maxRounds; round++) {
        Solution best = cur;
        vector<int> cand;

        // 1. 点交换：调整任意两个任务的位置。
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                cand = cur.perm;
                swap(cand[i], cand[j]);
                Solution s = evaluatePerm(ins, cand);
                if (s.F + 1e-9 < best.F) best = s;
            }
        }

        // 2. 区间反转：改变一段任务的先后顺序。
        for (int l = 0; l < k; l++) {
            for (int r = l + 1; r < k; r++) {
                cand = cur.perm;
                reverse(cand.begin() + l, cand.begin() + r + 1);
                Solution s = evaluatePerm(ins, cand);
                if (s.F + 1e-9 < best.F) best = s;
            }
        }

        // 3. 单点重插入：把一个任务抽出后插入到另一个位置。
        for (int i = 0; i < k; i++) {
            for (int pos = 0; pos < k; pos++) {
                if (pos == i) continue;
                cand = cur.perm;
                int item = cand[i];
                cand.erase(cand.begin() + i);
                int insertPos = pos;
                if (insertPos > i) insertPos--;
                cand.insert(cand.begin() + insertPos, item);
                Solution s = evaluatePerm(ins, cand);
                if (s.F + 1e-9 < best.F) best = s;
            }
        }

        // 4. 双载荷块交换与块重插入：直接改变“配送趟”的顺序，
        //    比单点扰动更符合双载荷问题结构。
        int blocks = (k + 1) / 2;
        for (int a = 0; a < blocks; a++) {
            int a1 = 2 * a;
            int lenA = min(2, k - a1);
            for (int b = a + 1; b < blocks; b++) {
                int b1 = 2 * b;
                int lenB = min(2, k - b1);
                if (lenA == lenB) {
                    cand = cur.perm;
                    for (int t = 0; t < lenA; t++) swap(cand[a1 + t], cand[b1 + t]);
                    Solution s = evaluatePerm(ins, cand);
                    if (s.F + 1e-9 < best.F) best = s;
                }
            }
        }

        // 5. 每个双载荷块内部翻转访问顺序。
        for (int pos = 0; pos + 1 < k; pos += 2) {
            cand = cur.perm;
            swap(cand[pos], cand[pos + 1]);
            Solution s = evaluatePerm(ins, cand);
            if (s.F + 1e-9 < best.F) best = s;
        }

        if (best.F + 1e-9 < cur.F) cur = best;
        else break;
    }
    return cur;
}

Solution largeNeighborhoodSearch(
    const Instance& ins,
    vector<int> start,
    const Config& cfg,
    mt19937& rng,
    double destroyRatio = -1.0,
    bool acceptWorse = true
) {
    if (destroyRatio < 0) destroyRatio = cfg.lnsDestroyRatio;

    // 多起点：LNS 从当前已知较好解开始，但额外比较几个结构化初始解，避免被 SA/EDD 单一起点限制。
    vector<vector<int>> starts;
    starts.push_back(start);
    starts.push_back(eddPerm(ins));
    starts.push_back(nearestGreedyPerm(ins));
    starts.push_back(matchingEDDPerm(ins));
    Solution cur; cur.F = numeric_limits<double>::infinity();
    for (auto& p : starts) {
        Solution s = evaluatePerm(ins, p);
        if (s.F < cur.F) cur = s;
    }
    cur = shortIntensification(ins, cur, max(10, cfg.lnsPolishIterations / 2), rng);
    Solution best = cur;

    uniform_real_distribution<double> ur(0.0, 1.0);
    double T = cfg.lnsT0;
    int baseRemove = max(2, (int)round(ins.k * destroyRatio));
    baseRemove = min(baseRemove, max(1, ins.k - 1));

    // 五类破坏算子：随机、最坏、相关、连续块、坏双载荷块。
    vector<double> weight = {1.0, 1.4, 1.4, 0.9, 1.2};
    vector<double> score(weight.size(), 0.0), used(weight.size(), 0.0);
    int noImprove = 0;

    for (int it = 0; it < cfg.lnsIterations; it++) {
        int op = rouletteSelect(weight, rng);
        used[op] += 1.0;

        // 破坏规模自适应：前期更大，后期更精细；偶尔大扰动跳出局部最优。
        double phase = (double)it / max(1, cfg.lnsIterations - 1);
        double ratioJitter = 0.75 + 0.65 * ur(rng);
        int removeCount = (int)round(baseRemove * (1.25 - 0.45 * phase) * ratioJitter);
        if (ur(rng) < 0.12) removeCount = max(removeCount, (int)round(ins.k * 0.45));
        removeCount = max(2, min(removeCount, max(1, ins.k - 1)));

        auto [base, removed] = destroySolution(ins, cur.perm, removeCount, op, rng);
        if (removed.empty()) continue;
        vector<int> candPerm = repairByRegretInsertion(ins, base, removed, rng);
        Solution nxt = evaluatePerm(ins, candPerm);

        // 对有潜力的候选解进行短局部强化，避免 LNS 只完成粗修复。
        if (nxt.F <= cur.F || ur(rng) < 0.35) {
            nxt = shortIntensification(ins, nxt, cfg.lnsPolishIterations, rng);
            if (nxt.F <= cur.F || ur(rng) < 0.18) nxt = deterministicPolish(ins, nxt, 1);
        }

        double delta = nxt.F - cur.F;
        bool accepted = false;
        if (delta <= 0 || (acceptWorse && ur(rng) < exp(-delta / max(1e-12, T)))) {
            cur = nxt;
            accepted = true;
        }

        if (nxt.F < best.F) {
            best = nxt;
            cur = nxt;
            score[op] += 9.0;
            noImprove = 0;
        } else if (accepted && delta < 0) {
            score[op] += 3.0;
            noImprove++;
        } else if (accepted) {
            score[op] += 0.8;
            noImprove++;
        } else {
            noImprove++;
        }

        // 周期性更新算子权重，形成简化版 ALNS。
        if ((it + 1) % cfg.lnsSegmentLength == 0) {
            for (int i = 0; i < (int)weight.size(); i++) {
                double avgScore = (used[i] > 0 ? score[i] / used[i] : 0.0);
                weight[i] = 0.80 * weight[i] + 0.20 * max(0.15, avgScore);
                score[i] = 0.0;
                used[i] = 0.0;
            }
        }

        // 长时间无提升时，从最优解附近重启，而不是继续在坏区域游走。
        if (noImprove >= cfg.lnsStagnationLimit) {
            cur = best;
            int kickRemove = max(2, min(ins.k - 1, (int)round(ins.k * 0.38)));
            auto [base2, rem2] = destroySolution(ins, cur.perm, kickRemove, 2, rng);
            if (!rem2.empty()) cur = evaluatePerm(ins, repairByRegretInsertion(ins, base2, rem2, rng));
            if (cur.F < best.F) best = cur;
            noImprove = 0;
            T = max(T, cfg.lnsT0 * 0.35);
        }

        T *= cfg.lnsAlpha;
    }

    // 最终强化：用短局部搜索收尾，通常能压掉修复造成的小瑕疵。
    best = shortIntensification(ins, best, cfg.lnsPolishIterations * 2, rng);
    best = deterministicPolish(ins, best, 3);
    return best;
}



// 多轮强化版 LNS：用不同破坏比例独立尝试，最后取最优。
Solution eliteLNS(const Instance& ins, vector<int> start, const Config& cfg, mt19937& rng) {
    Solution seed = evaluatePerm(ins, start);
    seed = deterministicPolish(ins, seed, 2);
    Solution best = seed;

    vector<double> ratios = {0.18, 0.30, 0.42};
    for (double ratio : ratios) {
        Solution cur = largeNeighborhoodSearch(ins, best.perm, cfg, rng, ratio, true);
        cur = deterministicPolish(ins, cur, 2);
        if (cur.F < best.F) best = cur;
    }
    return best;
}

Solution exactOptimalByPermutation(const Instance& ins) {
    vector<int> p(ins.k);
    iota(p.begin(), p.end(), 0);
    Solution best; best.F = numeric_limits<double>::infinity();
    do {
        Solution cur = evaluatePerm(ins, p);
        if (cur.F < best.F) best = cur;
    } while (next_permutation(p.begin(), p.end()));
    return best;
}

template <class Func>
Solution timedRun(Func f, double& ms) {
    auto st = chrono::steady_clock::now();
    Solution sol = f();
    auto ed = chrono::steady_clock::now();
    ms = chrono::duration<double, milli>(ed - st).count();
    return sol;
}

void addRun(RunValues& rv, const Solution& sol, int ref, double ms) {
    rv.F.push_back(sol.F);
    rv.D.push_back(sol.D);
    rv.tard.push_back(sol.tardiness);
    rv.gap.push_back((sol.F - ref) / max(1, ref) * 100.0);
    rv.timeMs.push_back(ms);
}

double improvementPct(double base, double improved) {
    if (fabs(base) < 1e-12) return 0.0;
    return (base - improved) / base * 100.0;
}

double gapPct(double value, double opt) {
    if (fabs(opt) < 1e-12) return 0.0;
    return (value - opt) / opt * 100.0;
}

void writeOverall(const string& fn, const map<string, Stats>& table) {
    ofstream fout(fn);
    fout << "方法,目标值,总距离,总迟到时间,距离参考差距百分比,平均时间毫秒,目标值标准差\n";
    for (auto& [name, st] : table) {
        fout << name << "," << fixed << setprecision(4)
             << st.F << "," << st.D << "," << st.tardiness << "," << st.gap << "," << st.timeMs << "," << st.stdF << "\n";
    }
}

void writeByK(const string& fn, const vector<tuple<int, Stats, Stats, Stats, Stats, Stats>>& rows) {
    ofstream fout(fn);
    fout << "目标数,EDD目标值,匹配EDD目标值,局部搜索目标值,模拟退火目标值,LNS目标值,LNS总迟到时间,LNS平均时间毫秒\n";
    for (auto row : rows) {
        int k; Stats edd, mat, loc, sa, lns;
        tie(k, edd, mat, loc, sa, lns) = row;
        fout << k << "," << fixed << setprecision(4)
             << edd.F << "," << mat.F << "," << loc.F << "," << sa.F << "," << lns.F << "," << lns.tardiness << "," << lns.timeMs << "\n";
    }
}

void writeByBeta(const string& fn, const vector<tuple<double, Stats, Stats, Stats, Stats, Stats>>& rows) {
    ofstream fout(fn);
    fout << "宽松系数,EDD目标值,匹配EDD目标值,局部搜索目标值,模拟退火目标值,LNS目标值,LNS总迟到时间\n";
    for (auto row : rows) {
        double beta; Stats edd, mat, loc, sa, lns;
        tie(beta, edd, mat, loc, sa, lns) = row;
        fout << fixed << setprecision(4)
             << beta << "," << edd.F << "," << mat.F << "," << loc.F << "," << sa.F << "," << lns.F << "," << lns.tardiness << "\n";
    }
}

void writeByLambda(const string& fn, const vector<tuple<double, Stats, Stats>>& rows) {
    ofstream fout(fn);
    fout << "惩罚系数,LNS总距离,LNS总迟到时间,LNS目标值,模拟退火目标值\n";
    for (auto row : rows) {
        double lambda; Stats lns, sa;
        tie(lambda, lns, sa) = row;
        fout << fixed << setprecision(4)
             << lambda << "," << lns.D << "," << lns.tardiness << "," << lns.F << "," << sa.F << "\n";
    }
}

void writeAblation(const string& fn, const map<string, Stats>& table) {
    ofstream fout(fn);
    fout << "方法,破坏比例,是否概率接受,目标值,总迟到时间,平均时间毫秒\n";
    vector<pair<string, string>> order = {{"LNS-小破坏", "0.15"}, {"LNS-标准", "0.28"}, {"LNS-大破坏", "0.45"}, {"LNS-无概率接受", "0.28"}};
    for (auto [name, ratio] : order) {
        auto it = table.find(name);
        if (it == table.end()) continue;
        string accept = (name == "LNS-无概率接受" ? "否" : "是");
        const Stats& st = it->second;
        fout << name << "," << ratio << "," << accept << "," << fixed << setprecision(4)
             << st.F << "," << st.tardiness << "," << st.timeMs << "\n";
    }
}

void writeExact(const string& fn, const vector<tuple<int, Stats, Stats, Stats, Stats, double, double, double>>& rows) {
    ofstream fout(fn);
    fout << "目标数,精确最优值,匹配EDD目标值,模拟退火目标值,LNS目标值,匹配EDD最优差距百分比,模拟退火最优差距百分比,LNS最优差距百分比\n";
    for (auto row : rows) {
        int k; Stats opt, mat, sa, lns; double gMat, gSA, gLNS;
        tie(k, opt, mat, sa, lns, gMat, gSA, gLNS) = row;
        fout << fixed << setprecision(4)
             << k << "," << opt.F << "," << mat.F << "," << sa.F << "," << lns.F << "," << gMat << "," << gSA << "," << gLNS << "\n";
    }
}

void writePaired(const string& fn, const vector<vector<double>>& rows) {
    ofstream fout(fn);
    fout << "实例编号,目标数,宽松系数,惩罚系数,匹配EDD目标值,局部搜索目标值,模拟退火目标值,LNS目标值,LNS相对模拟退火提升百分比,LNS相对局部搜索提升百分比,LNS相对匹配EDD提升百分比\n";
    for (auto r : rows) {
        fout << fixed << setprecision(4)
             << (int)r[0] << "," << (int)r[1] << "," << r[2] << "," << r[3] << ","
             << r[4] << "," << r[5] << "," << r[6] << "," << r[7] << "," << r[8] << "," << r[9] << "," << r[10] << "\n";
    }
}

void printLine(const string& name, const Stats& st) {
    cout << left << setw(18) << name << " F=" << setw(10) << fixed << setprecision(2) << st.F
         << " D=" << setw(8) << st.D << " late=" << setw(8) << st.tardiness
         << " time(ms)=" << setw(8) << st.timeMs << " std=" << st.stdF << "\n";
}

struct AlgorithmsResult { Solution edd, near, mat, loc, sa, lns; double tEdd=0,tNear=0,tMat=0,tLoc=0,tSA=0,tLNS=0; };

AlgorithmsResult runAll(const Instance& ins, const Config& cfg, mt19937& rng) {
    AlgorithmsResult r;
    r.edd = timedRun([&]{ return evaluatePerm(ins, eddPerm(ins)); }, r.tEdd);
    r.near = timedRun([&]{ return evaluatePerm(ins, nearestGreedyPerm(ins)); }, r.tNear);
    r.mat = timedRun([&]{ return evaluatePerm(ins, matchingEDDPerm(ins)); }, r.tMat);
    r.loc = timedRun([&]{ return localSearch(ins, eddPerm(ins), cfg.localIterations, rng); }, r.tLoc);
    r.sa = timedRun([&]{ return simulatedAnnealing(ins, (rng()%2?eddPerm(ins):randomPerm(ins.k,rng)), cfg, rng); }, r.tSA);
    vector<int> start = r.sa.F < r.mat.F ? r.sa.perm : r.mat.perm;
    r.lns = timedRun([&]{ return eliteLNS(ins, start, cfg, rng); }, r.tLNS);
    return r;
}

int runAllExperiments() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.setf(ios::unitbuf);
    Config cfg;
    filesystem::create_directories("results/csv");
    mt19937 rng(cfg.seed);
    cout << "===== Larger-k LNS Double-load Grid Delivery Experiment =====\n";
    cout << "Grid: " << cfg.n << "x" << cfg.m << ", obstacleRate=" << cfg.obstacleRate << "\n";
    cout << "Program started. Running experiments...\n";

    cout << "[1/8] 总体实验开始...\n";
    // 表1：总体实验
    {
        int k = 32; double beta = 1.6, lambda = 1.0;
        map<string, RunValues> rv;
        for (int inst = 0; inst < cfg.instancesPerGroup; inst++) {
            Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
            int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
            for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                auto r = runAll(ins, cfg, rng);
                addRun(rv["EDD贪心"], r.edd, ref, r.tEdd);
                addRun(rv["最近邻贪心"], r.near, ref, r.tNear);
                addRun(rv["匹配EDD"], r.mat, ref, r.tMat);
                addRun(rv["局部搜索"], r.loc, ref, r.tLoc);
                addRun(rv["模拟退火"], r.sa, ref, r.tSA);
                addRun(rv["LNS"], r.lns, ref, r.tLNS);
            }
        }
        map<string, Stats> table;
        vector<string> order = {"EDD贪心", "最近邻贪心", "匹配EDD", "局部搜索", "模拟退火", "LNS"};
        cout << "\n总体实验：\n";
        for (auto& name : order) { table[name] = summarize(rv[name]); printLine(name, table[name]); }
        writeOverall("results/csv/table1_overall.csv", table);
    }

    cout << "[2/8] 配对统计实验开始...\n";
    // 表8：配对统计数据
    {
        int k = 32; double beta = 1.6, lambda = 1.0;
        vector<vector<double>> rows;
        for (int inst = 0; inst < cfg.instancesPerGroup; inst++) {
            Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
            RunValues matRV, locRV, saRV, lnsRV;
            int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
            for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                auto r = runAll(ins, cfg, rng);
                addRun(matRV, r.mat, ref, r.tMat);
                addRun(locRV, r.loc, ref, r.tLoc);
                addRun(saRV, r.sa, ref, r.tSA);
                addRun(lnsRV, r.lns, ref, r.tLNS);
            }
            Stats mat = summarize(matRV), loc = summarize(locRV), sa = summarize(saRV), lns = summarize(lnsRV);
            rows.push_back({(double)inst, (double)k, beta, lambda, mat.F, loc.F, sa.F, lns.F,
                            improvementPct(sa.F, lns.F), improvementPct(loc.F, lns.F), improvementPct(mat.F, lns.F)});
        }
        writePaired("results/csv/table8_paired_results.csv", rows);
    }

    cout << "[3/8] 不同目标数实验开始...\n";
    // 表2：不同目标数（重点观察 k 增大后的 LNS 优势）
    {
        vector<tuple<int, Stats, Stats, Stats, Stats, Stats>> rows;
        for (int k : cfg.kValues) {
            double beta = 1.6, lambda = 1.0;
            map<string, RunValues> rv;
            for (int inst = 0; inst < cfg.instancesPerGroup; inst++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    auto r = runAll(ins, cfg, rng);
                    addRun(rv["EDD"], r.edd, ref, r.tEdd);
                    addRun(rv["MAT"], r.mat, ref, r.tMat);
                    addRun(rv["LOC"], r.loc, ref, r.tLoc);
                    addRun(rv["SA"], r.sa, ref, r.tSA);
                    addRun(rv["LNS"], r.lns, ref, r.tLNS);
                }
            }
            rows.push_back({k, summarize(rv["EDD"]), summarize(rv["MAT"]), summarize(rv["LOC"]), summarize(rv["SA"]), summarize(rv["LNS"])});
        }
        writeByK("results/csv/table2_by_k.csv", rows);
    }

    cout << "[4/8] 不同期限宽松系数实验开始...\n";
    // 表3：不同期限宽松系数
    {
        vector<tuple<double, Stats, Stats, Stats, Stats, Stats>> rows;
        int k = 32; double lambda = 1.0;
        for (double beta : cfg.betaValues) {
            map<string, RunValues> rv;
            for (int inst = 0; inst < cfg.instancesPerGroup; inst++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    auto r = runAll(ins, cfg, rng);
                    addRun(rv["EDD"], r.edd, ref, r.tEdd);
                    addRun(rv["MAT"], r.mat, ref, r.tMat);
                    addRun(rv["LOC"], r.loc, ref, r.tLoc);
                    addRun(rv["SA"], r.sa, ref, r.tSA);
                    addRun(rv["LNS"], r.lns, ref, r.tLNS);
                }
            }
            rows.push_back({beta, summarize(rv["EDD"]), summarize(rv["MAT"]), summarize(rv["LOC"]), summarize(rv["SA"]), summarize(rv["LNS"])});
        }
        writeByBeta("results/csv/table3_by_beta.csv", rows);
    }

    cout << "[5/8] 不同惩罚系数实验开始...\n";
    // 表4：不同惩罚系数
    {
        vector<tuple<double, Stats, Stats>> rows;
        int k = 32; double beta = 1.6;
        for (double lambda : cfg.lambdaValues) {
            map<string, RunValues> rv;
            for (int inst = 0; inst < cfg.instancesPerGroup; inst++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    auto r = runAll(ins, cfg, rng);
                    addRun(rv["SA"], r.sa, ref, r.tSA);
                    addRun(rv["LNS"], r.lns, ref, r.tLNS);
                }
            }
            rows.push_back({lambda, summarize(rv["LNS"]), summarize(rv["SA"])});
        }
        writeByLambda("results/csv/table4_by_lambda.csv", rows);
    }

    cout << "[6/8] LNS 消融实验开始...\n";
    // 表5：LNS 消融
    {
        int k = 32; double beta = 1.6, lambda = 1.0;
        map<string, RunValues> rv;
        for (int inst = 0; inst < cfg.instancesPerGroup; inst++) {
            Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
            int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
            for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                double ms = 0;
                vector<int> start = simulatedAnnealing(ins, eddPerm(ins), cfg, rng).perm;
                Solution a = timedRun([&]{ return largeNeighborhoodSearch(ins, start, cfg, rng, 0.15, true); }, ms); addRun(rv["LNS-小破坏"], a, ref, ms);
                Solution b = timedRun([&]{ return largeNeighborhoodSearch(ins, start, cfg, rng, 0.28, true); }, ms); addRun(rv["LNS-标准"], b, ref, ms);
                Solution c = timedRun([&]{ return largeNeighborhoodSearch(ins, start, cfg, rng, 0.45, true); }, ms); addRun(rv["LNS-大破坏"], c, ref, ms);
                Solution d = timedRun([&]{ return largeNeighborhoodSearch(ins, start, cfg, rng, 0.28, false); }, ms); addRun(rv["LNS-无概率接受"], d, ref, ms);
            }
        }
        map<string, Stats> table;
        for (auto name : {"LNS-小破坏", "LNS-标准", "LNS-大破坏", "LNS-无概率接受"}) table[name] = summarize(rv[name]);
        writeAblation("results/csv/table5_ablation.csv", table);
    }

    cout << "[7/8] 扩展规模实验开始...\n";
    // 表6：扩展规模（较大 k 主实验）
    {
        vector<tuple<int, Stats, Stats, Stats, Stats, Stats>> rows;
        double beta = 1.6, lambda = 1.0;
        for (int k : cfg.extendedKValues) {
            map<string, RunValues> rv;
            for (int inst = 0; inst < cfg.extendedInstancesPerGroup; inst++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
                for (int rep = 0; rep < cfg.repeatsPerInstance; rep++) {
                    auto r = runAll(ins, cfg, rng);
                    addRun(rv["EDD"], r.edd, ref, r.tEdd);
                    addRun(rv["MAT"], r.mat, ref, r.tMat);
                    addRun(rv["LOC"], r.loc, ref, r.tLoc);
                    addRun(rv["SA"], r.sa, ref, r.tSA);
                    addRun(rv["LNS"], r.lns, ref, r.tLNS);
                }
            }
            rows.push_back({k, summarize(rv["EDD"]), summarize(rv["MAT"]), summarize(rv["LOC"]), summarize(rv["SA"]), summarize(rv["LNS"])});
        }
        writeByK("results/csv/table6_extended_k.csv", rows);
    }

    cout << "[8/8] 小规模精确最优对比开始...\n";
    // 表7：小规模精确最优对比
    {
        vector<tuple<int, Stats, Stats, Stats, Stats, double, double, double>> rows;
        double beta = 1.6, lambda = 1.0;
        for (int k : cfg.exactKValues) {
            if (k > cfg.exactPermutationLimit) continue;
            RunValues optRV, matRV, saRV, lnsRV;
            vector<double> gMat, gSA, gLNS;
            for (int inst = 0; inst < cfg.exactInstancesPerGroup; inst++) {
                Instance ins = generateInstance(cfg.n, cfg.m, k, cfg.obstacleRate, beta, lambda, rng);
                int ref = distanceReferenceMatching(ins, cfg.exactMatchingLimit).lb;
                double ms = 0;
                Solution opt = timedRun([&]{ return exactOptimalByPermutation(ins); }, ms); addRun(optRV, opt, ref, ms);
                Solution mat = timedRun([&]{ return evaluatePerm(ins, matchingEDDPerm(ins)); }, ms); addRun(matRV, mat, ref, ms); gMat.push_back(gapPct(mat.F, opt.F));
                Solution sa = timedRun([&]{ return simulatedAnnealing(ins, eddPerm(ins), cfg, rng); }, ms); addRun(saRV, sa, ref, ms); gSA.push_back(gapPct(sa.F, opt.F));
                Solution lns = timedRun([&]{ return eliteLNS(ins, sa.perm, cfg, rng); }, ms); addRun(lnsRV, lns, ref, ms); gLNS.push_back(gapPct(lns.F, opt.F));
            }
            rows.push_back({k, summarize(optRV), summarize(matRV), summarize(saRV), summarize(lnsRV), meanOf(gMat), meanOf(gSA), meanOf(gLNS)});
        }
        writeExact("results/csv/table7_exact_gap.csv", rows);
    }

    cout << "\n全部实验完成。CSV 文件已生成。\n";
    return 0;
}

int main() {
    try {
        return runAllExperiments();
    } catch (const std::exception& e) {
        ofstream ferr("run_log.txt", ios::app);
        ferr << "\n程序发生异常: " << e.what() << "\n";
        cerr << "程序发生异常: " << e.what() << "\n";
        cerr << "请把 run_log.txt 发给我定位。\n";
        return 1;
    } catch (...) {
        ofstream ferr("run_log.txt", ios::app);
        ferr << "\n程序发生未知异常。\n";
        cerr << "程序发生未知异常，请把 run_log.txt 发给我定位。\n";
        return 1;
    }
}
