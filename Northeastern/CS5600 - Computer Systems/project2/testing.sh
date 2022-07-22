# Make sure dbserver is running before running this test.

# Test with a single thread
echo "Set key0 to value0"
./dbtest -t 1 -S key0 value0
echo "Read key0"
./dbtest -t 1 -G key0

echo "Modify key0 to value0_modified_111111"
./dbtest -t 1 -S key0 "value0_modified_111111"
echo "Read key0"
./dbtest -t 1 -G key0

echo "Modify key0 to value0_modified_111111_22222222222222222"
./dbtest -t 1 -S key0 "value0_modified_111111_22222222222222222"
echo "Read key0"
./dbtest -t 1 -G key0

echo "Delete key0"
./dbtest -t -1 -D key0
echo "Read key0"
./dbtest -t 1 -G key0

echo "Test multiple threads with -T"
./dbtest -T

echo "Test multiple threads with 100 requests and 5 threads."
./dbtest -n 100 -t 5

echo "Test quit command."
./dbtest -q
