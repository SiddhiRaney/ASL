class Solution {
public:
    static string robotWithString(string& s) {
        int n = s.size();
        string res, stk, sufMin(n, 'z');
        
        for (int i = n - 2; i >= 0; --i)
            sufMin[i] = min(s[i], sufMin[i + 1] = s[i + 1]);

        for (int i = 0; i < n; ++i) {
            stk += s[i];
            while (!stk.empty() && (i == n - 1 || stk.back() <= sufMin[i + 1]))
                res += stk.back(), stk.pop_back();
        }
        return res;
    }
};
