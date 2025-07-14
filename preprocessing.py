class Solution {
public:
    static string robotWithString(string& s) {
        int n = s.length();
        string result, stack;
        vector<char> minRight(n + 1, '{'); // '{' > 'z'

        // Compute suffix minimum characters
        for (int i = n - 1; i >= 0; --i)
            minRight[i] = min(s[i], minRight[i + 1]);

        // Simulate robot push/pop
        for (int i = 0; i < n; ++i) {
            stack.push_back(s[i]);
            while (!stack.empty() && stack.back() <= minRight[i + 1]) {
                result.push_back(stack.back());
                stack.pop_back();
            }
        }
        return result;
    }
};
