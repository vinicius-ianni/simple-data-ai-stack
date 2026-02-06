"""TDD tests: Verify Monty sandbox security boundaries for healthcare data."""

import pydantic_monty
import pytest


class TestMontySecurityBoundaries:
    """Monty must block all unsafe operations — critical for HIPAA-like environments."""

    def test_blocks_file_access(self):
        """Sandboxed code cannot open files."""
        m = pydantic_monty.Monty('open("/etc/passwd", "r").read()')
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="open"):
            m.run()

    def test_blocks_os_operations(self):
        """Sandboxed code cannot use os module for system access."""
        m = pydantic_monty.Monty('import os\nos.listdir(".")')
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="listdir"):
            m.run()

    def test_blocks_subprocess(self):
        """Sandboxed code cannot import subprocess."""
        m = pydantic_monty.Monty('__import__("subprocess").run(["ls"])')
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="__import__"):
            m.run()

    def test_blocks_exec(self):
        """Sandboxed code cannot use exec()."""
        m = pydantic_monty.Monty('exec("import subprocess")')
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="exec"):
            m.run()

    def test_blocks_eval(self):
        """Sandboxed code cannot use eval()."""
        m = pydantic_monty.Monty('eval("__import__(\'os\')")')
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="eval"):
            m.run()


class TestMontySafeExecution:
    """Monty must correctly execute safe data-processing code."""

    def test_basic_arithmetic(self):
        m = pydantic_monty.Monty("x + 1", inputs=["x"])
        assert m.run(inputs={"x": 41}) == 42

    def test_dict_processing(self):
        code = """
result = {"count": len(records), "first_id": records[0]["id"]}
result
"""
        m = pydantic_monty.Monty(code, inputs=["records"])
        result = m.run(inputs={"records": [{"id": "P-1"}, {"id": "P-2"}]})
        assert result == {"count": 2, "first_id": "P-1"}

    def test_list_comprehension(self):
        code = """
[r["name"] for r in records if "name" in r]
"""
        m = pydantic_monty.Monty(code, inputs=["records"])
        result = m.run(inputs={"records": [{"name": "A"}, {"x": 1}, {"name": "B"}]})
        assert result == ["A", "B"]

    def test_function_definition_and_call(self):
        """Our pipeline pattern: define process() then call it."""
        code = """
def process(records):
    valid = [r for r in records if "name" in r]
    return {"valid_count": len(valid)}

process(records)
"""
        m = pydantic_monty.Monty(code, inputs=["records"])
        result = m.run(inputs={"records": [{"name": "A"}, {"bad": 1}]})
        assert result == {"valid_count": 1}

    def test_string_manipulation(self):
        """De-identification requires string ops."""
        code = """
ssn = "123-45-6789"
masked = "***-**-" + ssn[-4:]
masked
"""
        m = pydantic_monty.Monty(code)
        assert m.run() == "***-**-6789"

    def test_print_callback_captures_output(self):
        captured = []

        def on_print(stream, text):
            captured.append(text)

        m = pydantic_monty.Monty('print("audit: step done")\n42')
        result = m.run(print_callback=on_print)
        assert result == 42
        assert "audit: step done" in "".join(captured)


class TestMontyErrorHandling:
    """Verify error types for bad code."""

    def test_syntax_error(self):
        with pytest.raises(pydantic_monty.MontySyntaxError):
            pydantic_monty.Monty("def incomplete(")

    def test_runtime_error_division_by_zero(self):
        m = pydantic_monty.Monty("1 / 0")
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="ZeroDivisionError"):
            m.run()

    def test_runtime_error_undefined_variable(self):
        m = pydantic_monty.Monty("undefined_var")
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="undefined_var"):
            m.run()
