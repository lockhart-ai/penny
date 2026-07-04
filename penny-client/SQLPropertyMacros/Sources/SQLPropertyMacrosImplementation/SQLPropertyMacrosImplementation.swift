import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public struct SQLPropertyMacro: PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        // Only handle variable declarations
        guard let varDecl = declaration.as(VariableDeclSyntax.self) else {
            return []
        }
        
        // Get the first binding pattern
        guard let binding = varDecl.bindings.first else {
            return []
        }
        
        // Get the property name
        guard let identifier = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text else {
            return []
        }
        
        // Get the type annotation
        guard let typeAnnotation = binding.typeAnnotation?.type else {
            return []
        }
        
        // Convert camelCase to snake_case
        let snakeCase = identifier.reduce("") { result, char in
            if char.isUppercase {
                return result + (result.isEmpty ? "" : "_") + String(char).lowercased()
            }
            return result + String(char)
        }
        
        // Create the expression property name
        let expressionName = "\(identifier)Exp"
        
        // Create the expression type based on the property type
        let expressionType = "SQLite.Expression<\(typeAnnotation)>"
        
        // Generate the expression property
        let expressionProperty = """
        fileprivate static var \(expressionName): \(expressionType) {
            Expression<\(typeAnnotation)>("\(snakeCase)")
        }
        """
        
        return [DeclSyntax(stringLiteral: expressionProperty)]
    }
}

@main
struct SQLPropertyMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        SQLPropertyMacro.self
    ]
}
