class Solution {
public:
    static string robotWithString(string& s) {
        int len = s.size();
        string result, stk;
        vector<char> minRight(len + 1, 'z' + 1);  // one extra space to avoid out-of-bound check

        // Precompute min character from i to end
        for (int i = len - 1; i >= 0; --i)
            minRight[i] = min(s[i], minRight[i + 1]);

        for (int i = 0; i < len; ++i) {
            stk.push_back(s[i]);

            // Pop from stack while top <= min of remaining characters
            while (!stk.empty() && stk.back() <= minRight[i + 1]) {
                result.push_back(stk.back());
                stk.pop_back();
            }
        }
        return result;
    }
};
