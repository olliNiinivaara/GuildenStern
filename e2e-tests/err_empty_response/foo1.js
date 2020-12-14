console.log("foo1");
var node = document.createElement("LI");
node.id = "foo1";
var textnode = document.createTextNode("foo1");
node.appendChild(textnode);
document.getElementById("ROOT").appendChild(node);
