class Solution {
public:
    static string robotWithString(string& s) {
        int len = s.size();
        string ans, stack, minRight(len, 'z');
        
        for (int i = len - 2; i >= 0; --i)
            minRight[i] = min(s[i], minRight[i + 1]);

        for (int i = 0; i < len; ++i) {
            stack += s[i];
            while (!stack.empty() && (i == len - 1 || stack.back() <= minRight[i + 1]))
                ans += stack.back(), stack.pop_back();
        }
        return ans;
    }
};
