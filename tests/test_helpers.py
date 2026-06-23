def assert_equal(result, expected):
    assert result == expected, f"Attendu : {expected}, obtenu : {result}"


def assert_true(result):
    assert result is True, f"Attendu : True, obtenu : {result}"


def assert_false(result):
    assert result is False, f"Attendu : False, obtenu : {result}"


def run_test(name, test_function):
    try:
        test_function()
        print("[OK]   " + name)
        return True
    except AssertionError as error:
        print("[FAIL] " + name)
        print("       " + str(error))
        return False
    except Exception as error:
        print("[ERROR] " + name)
        print("        " + type(error).__name__ + " : " + str(error))
        return False
