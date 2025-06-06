char stackBuffer[100000], suffixMinChar[100000];
int stackTop;

class Solution {
public:
    static string robotWithString(string& input) {
        int n = input.size();
        
        // Compute suffix min characters
        suffixMinChar[n - 1] = input[n - 1];
        for (int i = n - 2; i >= 0; --i)
            suffixMinChar[i] = min(input[i], suffixMinChar[i + 1]);

        string output;
        output.reserve(n);
        stackTop = -1;

        for (int i = 0; i < n; ++i) {
            stackBuffer[++stackTop] = input[i]; // Push onto stack

            // Pop while stack top <= min in suffix
            while (stackTop >= 0 && (i == n - 1 || stackBuffer[stackTop] <= suffixMinChar[i + 1])) {
                output += stackBuffer[stackTop--]; // Append to result
            }
        }

        return output;
    }
};
