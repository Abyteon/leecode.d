class Solution:
    def maximumSubarraySum(self, nums, k):
        """Find the maximum subarray sum of length k with distinct elements."""

        if len(nums) < k:
            return -1

        left = 0
        cur_sum = 0
        max_sum = 0
        window = {}

        for right, num in enumerate(nums):
            if num in window:
                # Move the left pointer to the right of the last occurrence of num
                left = max(left, window[num] + 1)

            window[num] = right

            if right - left + 1 == k:
                # If the window size is k, calculate the sum
                cur_sum = sum(nums[left : right + 1])
                max_sum = max(max_sum, cur_sum)
                left += 1
        return max_sum


if __name__ == "__main__":
    nums = [1, 5, 4, 2, 9, 9, 9]
    nums = [4, 4, 4]
    print(nums)
    solution = Solution()
    res = solution.maximumSubarraySum(nums, 3)
    print(res)  # Expected output: 16
