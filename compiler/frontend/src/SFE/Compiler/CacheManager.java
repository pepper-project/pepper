package SFE.Compiler;

import java.util.HashMap;

import SFE.Compiler.Operators.StructAccessOperator;

public class CacheManager {
	private static HashMap<Integer, LvalExpression> notCached = new HashMap<Integer, LvalExpression>();
	private static HashMap<Integer, LvalExpression> memoryMapping = new HashMap<Integer, LvalExpression>();

	public static void setMemoryMapping(LvalExpression lvalue) {
		if (lvalue.getType() instanceof StructType) {
			StructType struct = (StructType) lvalue.getDeclaredType();
			for (int i = 0; i < struct.getFields().size(); i++) {
				String fieldname = struct.getFields().get(i);
				LvalExpression component = new StructAccessOperator(fieldname)
				    .resolve(lvalue);
				setMemoryMapping(component);
			}
		} else {
			// either array type or simple type.
			for (int i = 0; i < lvalue.size(); i++) {
				memoryMapping.put(lvalue.fieldEltAt(i).getAddress(),
				    lvalue.fieldEltAt(i));
			}
		}
	}

	public static void addCache(int address) {
		notCached.remove(address);
	}

	public static boolean isCached(int address) {
		if (notCached.containsKey(address)) {
			return false;
		} else {
			return true;
		}
	}

	// write cached variable at address back to RAM.
	public static Statement invalidateCache(int addr) {
		BlockStatement result = new BlockStatement();

		if (!notCached.containsKey(addr)) {
			LvalExpression lval = memoryMapping.get(addr);
			if (lval != null) {
				// the result is cached somewhere, generate ramput
				RamPutEnhancedStatement ramput = new RamPutEnhancedStatement(
				    IntConstant.valueOf(addr), lval, IntConstant.ONE, true);
				notCached.put(addr, lval);
				result.addStatement(ramput);
			} else {
				// throw new RuntimeException("Assertion failure.");
				System.out.println("WARNING: potential inaccurate point-to analysis.");
			}
		}

		return result;
	}
}
