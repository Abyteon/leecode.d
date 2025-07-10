class Solution:
    def maxScore(self, cardPoints: list[int], k: int) -> int:
        current_sum = sum(cardPoints[:k])
        max_sum = current_sum

        for i in range(k):
            current_sum = current_sum - cardPoints[k - 1 - i] + cardPoints[-1 - i]
            max_sum = max(max_sum, current_sum)
        return max_sum


if __name__ == "__main__":
    solution = Solution()
    print(solution.maxScore([1, 2, 3, 4, 5, 6, 1], 3))  # Output: 12
