#include <catch2/catch_all.hpp>
// Include catch2 first.
#include "array.h"
#include "cuda_helpers.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/atomic64_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/compact_acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.h"
#include "hash_table_test_cuda.h"
#include "timings.h"

static constexpr uint32_t itemSizeInU32 = 2;

template <typename A, typename B>
static bool compareItem(A lhs, B rhs)
{
    for (uint32_t i = 0; i < itemSizeInU32; ++i) {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

const char* TAGS = "[Atomic64HashTable][AccelerationHashTable][CompactAccelerationHashTable][TicketBoardHashTable]";
#define HASH_TABLES_CPU Atomic64HashTable<EMemoryType::CPU>, AccelerationHashTable<EMemoryType::CPU>, CompactAccelerationHashTable<EMemoryType::CPU>, TicketBoardHashTable<EMemoryType::CPU>
#define HASH_TABLES_GPU Atomic64HashTable<EMemoryType::GPU_Malloc>, AccelerationHashTable<EMemoryType::GPU_Malloc>, CompactAccelerationHashTable<EMemoryType::GPU_Malloc>

TEMPLATE_TEST_CASE("GPUHashTable CPU", TAGS, HASH_TABLES_CPU)
{
    constexpr size_t numInsertItems = 1024;
    constexpr size_t numNonInsertItems = 1024;
    constexpr size_t numItems = numInsertItems + numNonInsertItems;
    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto myHashTable = TestType::allocate(3, numItems, itemSizeInU32);

    std::vector<uint32_t> insertedLocations(numInsertItems);
    for (uint32_t i = 0; i < numInsertItems; ++i) {
        const uint32_t* pItem = &items[i * itemSizeInU32];
        insertedLocations[i] = myHashTable.add(pItem);
        const auto cmp = myHashTable.decodePointer(insertedLocations[i]);
        REQUIRE(compareItem(cmp, pItem));

        const auto lookUpLocation = myHashTable.find(pItem);
        CHECK(lookUpLocation == insertedLocations[i]);
    }

    for (size_t i = 0; i < numItems; ++i) {
        const uint32_t* item = &items[i * itemSizeInU32];
        const auto lookUpLocation = myHashTable.find(item);

        if (i < numInsertItems) {
            const auto expectedLocation = insertedLocations[i];
            CHECK(lookUpLocation != myHashTable.not_found);
            CHECK(lookUpLocation == expectedLocation);
        }

        if (lookUpLocation != myHashTable.not_found) {
            const auto storedItem = myHashTable.decodePointer(lookUpLocation);
            CHECK(compareItem(storedItem, item));
        }
    }

    myHashTable.free();
}

TEMPLATE_TEST_CASE("GPUHashTable CPU construction; GPU decoding (thread)", TAGS, HASH_TABLES_CPU)
{
    constexpr size_t numInsertItems = 1024;
    constexpr size_t numNonInsertItems = 1024;
    constexpr size_t numItems = numInsertItems + numNonInsertItems;
    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto myHashTable = TestType::allocate(3, numItems, itemSizeInU32);

    std::vector<uint32_t> insertedLocations(numInsertItems);
    for (uint32_t i = 0; i < numInsertItems; ++i) {
        insertedLocations[i] = myHashTable.add(&items[i * itemSizeInU32]);
    }

    auto myHashTableGPU = myHashTable.template copy<EMemoryType::GPU_Malloc>();

    auto itemsGPU = StaticArray<uint32_t>::allocate("items", items, EMemoryType::GPU_Malloc);
    auto lookUpLocationsGPU = StaticArray<uint32_t>::allocate("out lookup locations", numItems, EMemoryType::GPU_Managed);

    find_cuda(myHashTableGPU, itemsGPU, lookUpLocationsGPU);

    for (size_t i = 0; i < numItems; ++i) {
        const auto* inItem = &items[i * itemSizeInU32];
        const auto lookUpLocation = lookUpLocationsGPU[i];

        if (i < numInsertItems) {
            CHECK(lookUpLocation != myHashTable.not_found);
            const auto expectedLocation = insertedLocations[i];
            CHECK(lookUpLocation == expectedLocation);

            const auto outItem = myHashTable.decodePointer(lookUpLocation);
            CHECK(compareItem(outItem, inItem));
        }

        if (lookUpLocation != myHashTable.not_found) {
            const auto outItem = myHashTable.decodePointer(lookUpLocation);
            CHECK(compareItem(outItem, inItem));
        }
    }

    itemsGPU.free();
    lookUpLocationsGPU.free();
    myHashTableGPU.free();
    myHashTable.free();
}
TEMPLATE_TEST_CASE("GPUHashTable CPU construction; GPU decoding (warp)", TAGS, HASH_TABLES_CPU)
{
    constexpr size_t numInsertItems = 1024;
    constexpr size_t numNonInsertItems = 1024;
    constexpr size_t numItems = numInsertItems + numNonInsertItems;
    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto myHashTable = TestType::allocate(3, numItems, itemSizeInU32);

    std::vector<uint32_t> insertedLocations(numInsertItems);
    for (uint32_t i = 0; i < numInsertItems; ++i) {
        insertedLocations[i] = myHashTable.add(&items[i * itemSizeInU32]);
    }

    auto myHashTableGPU = myHashTable.template copy<EMemoryType::GPU_Malloc>();

    auto itemsGPU = StaticArray<uint32_t>::allocate("items", items, EMemoryType::GPU_Malloc);
    auto lookUpLocationsGPU = StaticArray<uint32_t>::allocate("out lookup locations", numItems, EMemoryType::GPU_Managed);

    findAsWarp_cuda(myHashTableGPU, itemsGPU, lookUpLocationsGPU);

    for (size_t i = 0; i < numItems; ++i) {
        const auto* inItem = &items[i * itemSizeInU32];
        const auto lookUpLocation = lookUpLocationsGPU[i];

        if (i < numInsertItems) {
            CHECK(lookUpLocation != myHashTable.not_found);
            const auto expectedLocation = insertedLocations[i];
            CHECK(lookUpLocation == expectedLocation);

            const auto outItem = myHashTable.decodePointer(lookUpLocation);
            CHECK(compareItem(outItem, inItem));
        }

        if (lookUpLocation != myHashTable.not_found) {
            const auto outItem = myHashTable.decodePointer(lookUpLocation);
            CHECK(compareItem(outItem, inItem));
        }
    }

    itemsGPU.free();
    lookUpLocationsGPU.free();
    myHashTableGPU.free();
    myHashTable.free();
}

TEMPLATE_TEST_CASE("GPUHashTable GPU construction (thread) & decoding (warp)", TAGS, HASH_TABLES_GPU)
{
    constexpr size_t numInsertItems = 16 * 1024;
    constexpr size_t numNonInsertItems = 16 * 1024;
    constexpr size_t numItems = numInsertItems + numNonInsertItems;

    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));
    auto itemsGPU = StaticArray<uint32_t>::allocate("items", items, EMemoryType::GPU_Malloc);

    auto myHashTable = TestType::allocate(1024, numItems + 100000, itemSizeInU32);
    auto insertLocations = StaticArray<uint32_t>::allocate("out insert locations", numInsertItems, EMemoryType::GPU_Managed);
    add_cuda(myHashTable, itemsGPU.span().subspan(0, numInsertItems * itemSizeInU32), insertLocations);

    auto lookUpLocations = StaticArray<uint32_t>::allocate("out lookup locations", numItems, EMemoryType::GPU_Managed);
    findAsWarp_cuda(myHashTable, itemsGPU, lookUpLocations);

    auto myHashTableCPU = myHashTable.template copy<EMemoryType::CPU>();
    for (uint32_t i = 0; i < numInsertItems; ++i) {
        CHECK(lookUpLocations[i] == insertLocations[i]);
        CHECK(lookUpLocations[i] != myHashTable.not_found);
        const uint32_t* inItem = &items[i * itemSizeInU32];
        const auto outItem = myHashTableCPU.decodePointer(lookUpLocations[i]);
        CHECK(compareItem(inItem, outItem));
    }
    uint32_t numFoundNonInserted = 0;
    for (uint32_t i = numInsertItems; i < numItems; ++i) {
        if (lookUpLocations[i] != myHashTable.not_found) {
            const uint32_t* inItem = &items[i * itemSizeInU32];
            const auto outItem = myHashTableCPU.decodePointer(lookUpLocations[i]);
            CHECK(compareItem(inItem, outItem));
            ++numFoundNonInserted;
        }
    }
    CHECK(numFoundNonInserted < numItems / 100);

    insertLocations.free();
    lookUpLocations.free();
    itemsGPU.free();
    myHashTable.free();
    myHashTableCPU.free();
}

TEMPLATE_TEST_CASE("GPUHashTable GPU construction (warp) & decoding (warp)", TAGS, HASH_TABLES_GPU)
{
    constexpr size_t numInsertItems = 16 * 1024;
    constexpr size_t numNonInsertItems = 16 * 1024;
    constexpr size_t numItems = numInsertItems + numNonInsertItems;

    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));
    auto itemsGPU = StaticArray<uint32_t>::allocate("items", items, EMemoryType::GPU_Malloc);

    auto myHashTable = TestType::allocate(64, numItems + 100000, itemSizeInU32);
    auto insertLocations = StaticArray<uint32_t>::allocate("out insert locations", numInsertItems, EMemoryType::GPU_Managed);
    addAsWarp_cuda(myHashTable, itemsGPU.span().subspan(0, numInsertItems * itemSizeInU32), insertLocations);

    auto lookUpLocations = StaticArray<uint32_t>::allocate("out lookup locations", numItems, EMemoryType::GPU_Managed);
    findAsWarp_cuda(myHashTable, itemsGPU, lookUpLocations);

    auto myHashTableCPU = myHashTable.template copy<EMemoryType::CPU>();
    for (uint32_t i = 0; i < numInsertItems; ++i) {
        CHECK(lookUpLocations[i] == insertLocations[i]);
        CHECK(lookUpLocations[i] != myHashTable.not_found);
        const uint32_t* inItem = &items[i * itemSizeInU32];
        const auto outItem = myHashTableCPU.decodePointer(lookUpLocations[i]);
        CHECK(compareItem(inItem, outItem));
    }
    uint32_t numFoundNonInserted = 0;
    for (uint32_t i = numInsertItems; i < numItems; ++i) {
        if (lookUpLocations[i] != myHashTable.not_found) {
            const uint32_t* inItem = &items[i * itemSizeInU32];
            const auto outItem = myHashTableCPU.decodePointer(lookUpLocations[i]);
            CHECK(compareItem(inItem, outItem));
            ++numFoundNonInserted;
        }
    }
    CHECK(numFoundNonInserted < numItems / 100);

    insertLocations.free();
    lookUpLocations.free();
    itemsGPU.free();
    myHashTable.free();
    myHashTableCPU.free();
}

TEMPLATE_TEST_CASE("GPUHashTable::currentLoadFactor()", TAGS, HASH_TABLES_CPU)
{
    constexpr size_t numItems = 1000;
    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto myHashTable = TestType::allocate(3, numItems, itemSizeInU32);
    CHECK(myHashTable.currentLoadFactor() == 0);

    for (uint32_t i = 0; i < numItems; ++i) {
        myHashTable.add(&items[i * itemSizeInU32]);
    }

    const auto loadFactor = myHashTable.currentLoadFactor();
    const double expectedLoadFactor = (double)numItems / (double)myHashTable.numBuckets();
    CHECK(loadFactor == Catch::Approx(expectedLoadFactor));

    myHashTable.free();
}

#if CAPTURE_MEMORY_STATS_SLOW
TEMPLATE_TEST_CASE("GPUHashTable Delete CPU", TAGS, HASH_TABLES_CPU)
{
    constexpr size_t numInsertItems = 1024;
    constexpr size_t numNonInsertItems = 512;
    constexpr size_t numItems = numInsertItems + numNonInsertItems;
    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto myHashTable = CompactAccelerationHashTable<EMemoryType::CPU>::allocate(64, numItems, itemSizeInU32);

    std::vector<uint32_t> insertedLocations(numItems);
    for (uint32_t i = 0; i < numItems; ++i) {
        insertedLocations[i] = myHashTable.add(&items[i * itemSizeInU32]);
    }

    const auto loadFactorBefore = myHashTable.currentLoadFactor();
    myHashTable.clearActiveFlags();
    std::for_each(std::begin(insertedLocations), std::begin(insertedLocations) + numInsertItems,
        [&](uint32_t loc) {
            myHashTable.markAsActive(loc);
        });
    const uint32_t numItemsFreed = myHashTable.freeInactiveItems();
    CHECK(numItemsFreed == numNonInsertItems);
    const auto loadFactorAfter = myHashTable.currentLoadFactor();
    CHECK(loadFactorAfter < loadFactorBefore);

    for (size_t i = 0; i < numItems; ++i) {
        const uint32_t* pItem = &items[i * itemSizeInU32];
        const auto lookUpLocation = myHashTable.find(pItem);

        if (i < numInsertItems) {
            const auto expectedLocation = insertedLocations[i];
            CHECK(lookUpLocation != myHashTable.not_found);
            CHECK(lookUpLocation == expectedLocation);
        }

        if (lookUpLocation != myHashTable.not_found) {
            const uint32_t* pStoredItem = myHashTable.decodePointer(lookUpLocation);
            CHECK(compareItem(pItem, pStoredItem));
        }
    }

    myHashTable.free();
}

TEMPLATE_TEST_CASE("GPUHashTable Delete GPU", TAGS, HASH_TABLES_GPU)
{
    constexpr size_t numInsertItems = 1024;
    constexpr size_t numNonInsertItems = 512;
    constexpr size_t numItems = numInsertItems + numNonInsertItems;
    const auto items = GENERATE(take(10, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto myHashTableGPU = CompactAccelerationHashTable<EMemoryType::GPU_Malloc>::allocate(64, numItems, itemSizeInU32);

    auto itemsGPU = StaticArray<uint32_t>::allocate("itemsGPU", items, EMemoryType::GPU_Malloc);
    auto insertedLocationsGPU = StaticArray<uint32_t>::allocate("locations", numItems, EMemoryType::GPU_Malloc);
    addAsWarp_cuda(myHashTableGPU, itemsGPU.span(), insertedLocationsGPU.span());
    CUDA_CHECK_ERROR();

    myHashTableGPU.clearActiveFlags();
    markActiveElements_cuda(myHashTableGPU, insertedLocationsGPU.span().subspan(0, numInsertItems));
    const uint32_t numItemsFreed = myHashTableGPU.freeInactiveItems();
    CHECK(numItemsFreed == numNonInsertItems);

    auto myHashTableCPU = myHashTableGPU.copy<EMemoryType::CPU>();
    const auto insertedLocations = insertedLocationsGPU.copy_to_cpu();
    for (size_t i = 0; i < numItems; ++i) {
        const uint32_t* pItem = &items[i * itemSizeInU32];
        const auto lookUpLocation = myHashTableCPU.find(pItem);

        if (i < numInsertItems) {
            const auto expectedLocation = insertedLocations[i];
            CHECK(lookUpLocation != myHashTableCPU.not_found);
            CHECK(lookUpLocation == expectedLocation);
        }

        if (lookUpLocation != myHashTableCPU.not_found) {
            const uint32_t* pStoredItem = myHashTableCPU.decodePointer(lookUpLocation);
            CHECK(compareItem(pItem, pStoredItem));
        }
    }

    itemsGPU.free();
    insertedLocationsGPU.free();
    myHashTableGPU.free();
    myHashTableCPU.free();
}
#endif
