console.log("foo2");
var node = document.createElement("LI");
var textnode = document.createTextNode("foo2");
node.appendChild(textnode);
document.getElementById("ROOT").appendChild(node);