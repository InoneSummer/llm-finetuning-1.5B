const parser = require("@babel/parser");

const code = process.argv[2];

if (!code) {
  console.log(JSON.stringify({
    valid: false,
    components: 0,
    hooks: 0,
    jsx_elements: 0,
    error: "No code provided"
  }));
  process.exit(1);
}

try {
  const ast = parser.parse(code, {
    sourceType: "module",
    plugins: ["jsx", "typescript"]
  });

  let components = 0;
  let hooks = 0;
  let jsx_elements = 0;

  function traverse(node) {
    if (!node || typeof node !== "object") return;

    // 컴포넌트 카운트 (함수 선언 + 화살표 함수)
    if (
      node.type === "FunctionDeclaration" ||
      node.type === "ArrowFunctionExpression"
    ) {
      components++;
    }

    // 훅 카운트 (use로 시작하는 함수 호출)
    if (
      node.type === "CallExpression" &&
      node.callee &&
      node.callee.name &&
      node.callee.name.startsWith("use")
    ) {
      hooks++;
    }

    // JSX 엘리먼트 카운트
    if (node.type === "JSXElement") {
      jsx_elements++;
    }

    for (const key of Object.keys(node)) {
      if (key !== "parent" && Array.isArray(node[key])) {
        node[key].forEach(traverse);
      } else if (key !== "parent") {
        traverse(node[key]);
      }
    }
  }

  traverse(ast);

  console.log(JSON.stringify({
    valid: true,
    components,
    hooks,
    jsx_elements,
    error: null
  }));

} catch (e) {
  console.log(JSON.stringify({
    valid: false,
    components: 0,
    hooks: 0,
    jsx_elements: 0,
    error: e.message
  }));
}
