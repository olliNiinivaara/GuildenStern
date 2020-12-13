console.log("foo1");
var node = document.createElement("LI");
var textnode = document.createTextNode("foo1");
node.appendChild(textnode);
document.getElementById("ROOT").appendChild(node);