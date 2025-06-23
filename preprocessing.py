class Solution {
public:
    static string robotWithString(string& s) {
        int len = s.size();
        string ans, stk;  // 'stk' used instead of 'stack' to avoid shadowing std::stack
        vector<char> minRight(len, 'z');

        // Precompute the minimum character to the right of each position
        for (int i = len - 2; i >= 0; --i)
            minRight[i] = min(s[i], minRight[i + 1]);

        // Traverse the string
        for (int i = 0; i < len; ++i) {
            stk += s[i];  // Push current character onto stk

            // Pop from stk if it's lexicographically smaller or equal to the smallest character remaining
            while (!stk.empty() && (i == len - 1 || stk.back() <= minRight[i + 1])) {
                ans += stk.back();
                stk.pop_back();
            }
        }
        return ans;
    }
};
