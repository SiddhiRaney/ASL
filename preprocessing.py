class Solution {
public:
    static string robotWithString(string& s) {
        int n = s.size();
        string result;
        result.reserve(n);

        // Suffix min character array
        vector<char> suffixMin(n);
        suffixMin[n - 1] = s[n - 1];
        for (int i = n - 2; i >= 0; --i)
            suffixMin[i] = min(s[i], suffixMin[i + 1]);

        // Simulate stack using string
        string stk;

        for (int i = 0; i < n; ++i) {
            stk.push_back(s[i]);
            // Pop from stack if top <= suffix min of remaining
            while (!stk.empty() && (i == n - 1 || stk.back() <= suffixMin[i + 1])) {
                result += stk.back();
                stk.pop_back();
            }
        }

        return result;
    }
};
